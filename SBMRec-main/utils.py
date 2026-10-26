# -*- coding: utf-8 -*-
"""Training and evaluation utilities for SBMRec."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def move_batch_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


class NegativeSampler:
    """Sample one positive followed by unseen negative item IDs per user."""

    def __init__(
        self,
        num_items: int,
        num_negatives: int,
        seed: int,
        user_seen_items: Optional[Sequence[Set[int]]] = None,
        exclude_seen: bool = True,
    ) -> None:
        if num_items <= 1:
            raise ValueError("num_items must be greater than one.")
        if num_negatives <= 0:
            raise ValueError("num_negatives must be positive.")
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)
        self.user_seen_items = user_seen_items
        self.exclude_seen = exclude_seen

    def _sample_for_user(self, positive: int, user_index: int) -> List[int]:
        forbidden: Set[int] = {0, positive}
        if (
            self.exclude_seen
            and self.user_seen_items is not None
            and 0 <= user_index < len(self.user_seen_items)
        ):
            forbidden.update(self.user_seen_items[user_index])

        available_count = (self.num_items - 1) - len(
            {item for item in forbidden if 1 <= item < self.num_items}
        )
        if available_count <= 0:
            forbidden = {0, positive}

        negatives: List[int] = []
        used: Set[int] = set()
        unique_possible = max(0, (self.num_items - 1) - len(forbidden))
        require_unique = unique_possible >= self.num_negatives

        attempts = 0
        max_attempts = max(1000, self.num_negatives * 100)
        while len(negatives) < self.num_negatives and attempts < max_attempts:
            attempts += 1
            candidate = int(self.rng.integers(1, self.num_items))
            if candidate in forbidden:
                continue
            if require_unique and candidate in used:
                continue
            negatives.append(candidate)
            used.add(candidate)

        if len(negatives) < self.num_negatives:
            # Deterministic fallback over the item range. Repetition is allowed
            # only when the catalog is too small for the requested sample size.
            pool = [
                item
                for item in range(1, self.num_items)
                if item not in forbidden
            ]
            if not pool:
                pool = [
                    item for item in range(1, self.num_items) if item != positive
                ]
            cursor = 0
            while len(negatives) < self.num_negatives:
                negatives.append(pool[cursor % len(pool)])
                cursor += 1
        return negatives

    def sample(
        self,
        positives: torch.Tensor,
        user_indices: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positives_cpu = positives.detach().cpu().tolist()
        users_cpu = user_indices.detach().cpu().tolist()
        candidates = []
        for positive, user_index in zip(positives_cpu, users_cpu):
            negatives = self._sample_for_user(int(positive), int(user_index))
            candidates.append([int(positive)] + negatives)
        candidate_tensor = torch.tensor(
            candidates, dtype=torch.long, device=device
        )
        positive_positions = torch.zeros(
            len(candidates), dtype=torch.long, device=device
        )
        return candidate_tensor, positive_positions


@dataclass
class RankingMetricAccumulator:
    ks: Sequence[int]
    sums: Dict[str, float] = field(default_factory=dict)
    count: int = 0

    def __post_init__(self) -> None:
        for k in self.ks:
            self.sums[f"HR@{k}"] = 0.0
            self.sums[f"NDCG@{k}"] = 0.0

    def update(self, logits: torch.Tensor, positive_index: int = 0) -> None:
        if logits.ndim != 2:
            raise ValueError("logits must have shape [B, C].")
        positive_scores = logits[:, positive_index : positive_index + 1]
        # Rank is 1 plus the number of candidates with a strictly higher score.
        ranks = (logits > positive_scores).sum(dim=1) + 1
        batch_size = logits.size(0)
        self.count += batch_size
        for k in self.ks:
            hits = ranks.le(k)
            self.sums[f"HR@{k}"] += float(hits.float().sum().item())
            discounts = torch.where(
                hits,
                1.0 / torch.log2(ranks.float() + 1.0),
                torch.zeros_like(ranks, dtype=torch.float),
            )
            self.sums[f"NDCG@{k}"] += float(discounts.sum().item())

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {name: float("nan") for name in self.sums}
        return {name: value / self.count for name, value in self.sums.items()}


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        improved = (
            self.best is None
            or (self.mode == "max" and value > self.best)
            or (self.mode == "min" and value < self.best)
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args: Dict,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": args,
        },
        output_path,
    )


def load_model_checkpoint(
    path: str,
    model: torch.nn.Module,
    device: torch.device,
) -> Dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def format_metrics(metrics: Dict[str, float]) -> str:
    return ", ".join(f"{name}: {value:.4f}" for name, value in metrics.items())
