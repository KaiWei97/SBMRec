#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate SBMRec on next item.

Each held-out target is ranked against 99 negatives sampled uniformly without
replacement from items the user never interacted with. The candidates are
sampled once with a fixed seed, saved to disk, and shared by all runs.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from data_loader import DatasetBundle, build_datasets, load_item_features
from models.sbmrec_model import SBMRec
from utils import (
    EarlyStopping,
    count_trainable_parameters,
    format_metrics,
    load_model_checkpoint,
    move_batch_to_device,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SBMRec"
    )

    data = parser.add_argument_group("data")
    data.add_argument("--data_root", type=str,
                      default=os.path.join(os.path.dirname(__file__), "data", "Kuaishou"))
    data.add_argument("--text_features", type=str, required=True,
                      help="Pre-extracted CLIP text features of the items.")
    data.add_argument("--image_features", type=str, required=True,
                      help="Pre-extracted CLIP image features of the items.")
    data.add_argument("--seq_len", type=int, default=50)
    data.add_argument("--min_user_interactions", type=int, default=5)
    data.add_argument("--min_favor_interactions", type=int, default=3)
    data.add_argument("--min_item_freq", type=int, default=1)
    data.add_argument("--click_behaviors", type=str, default="click")
    data.add_argument("--favor_behaviors", type=str, default="favor")
    data.add_argument("--no_cache", action="store_true")
    data.add_argument("--assume_chronological", action="store_true",
                      help="Allow a single CSV without timestamps whose rows are already in time order.")
    data.add_argument("--num_workers", type=int, default=0)

    model = parser.add_argument_group("model")
    model.add_argument("--d_model", type=int, default=256)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--alignment_weight", type=float, default=0.1,
                       help="Weight lambda of the alignment loss (Eq. 20).")
    model.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature tau of the alignment loss (Eq. 19).")
    model.add_argument("--filter_sharing", choices=("pair", "behavior", "modality", "shared"),
                       default="pair", help="Filter-sharing strategy (Table 3).")
    model.add_argument("--pooling", choices=("attention", "mean", "last"), default="attention",
                       help="Temporal pooling strategy (Table 5).")
    model.add_argument("--no_fourier", action="store_true", help="w/o BMSM-F.")
    model.add_argument("--no_wavelet", action="store_true", help="w/o BMSM-W.")
    model.add_argument("--no_calibration", action="store_true", help="w/o BMCT-C.")
    model.add_argument("--no_transfer", action="store_true", help="w/o BMCT-T.")
    model.add_argument("--no_adaptive_gate", action="store_true",
                       help="Equal-weight click/favorite fusion (used with --no_calibration for w/o BMCT).")

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--batch_size", type=int, default=1024)
    optim.add_argument("--epochs", type=int, default=100)
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument("--weight_decay", type=float, default=0.0)
    optim.add_argument("--num_neg", type=int, default=99, help="Training negatives per target.")
    optim.add_argument("--num_eval_neg", type=int, default=99, help="Evaluation negatives per target.")
    optim.add_argument("--eval_seed", type=int, default=2026,
                       help="Seed of the shared evaluation candidates.")
    optim.add_argument("--grad_clip", type=float, default=5.0)
    optim.add_argument("--eval_every", type=int, default=1)
    optim.add_argument("--patience", type=int, default=10)
    optim.add_argument("--amp", action="store_true")
    optim.add_argument("--seed", type=int, default=2088)
    optim.add_argument("--deterministic", action="store_true")
    optim.add_argument("--output_dir", type=str,
                       default=os.path.join(os.path.dirname(__file__), "checkpoints", "sbmrec"))

    args = parser.parse_args()
    if args.seq_len <= 0 or args.seq_len % 2:
        parser.error("--seq_len must be a positive even number for the Haar DWT.")
    if min(args.batch_size, args.d_model, args.epochs, args.eval_every, args.patience) <= 0:
        parser.error("Batch size, dimension, epochs, evaluation interval and patience must be positive.")
    if args.num_neg < 1 or args.num_eval_neg < 19:
        parser.error("Use at least one training negative and 19 evaluation negatives (for HR@20).")
    if not math.isfinite(args.alignment_weight) or args.alignment_weight < 0:
        parser.error("--alignment_weight must be finite and nonnegative.")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive.")
    return args


# ---------------------------------------------------------------------------
# Negative sampling and evaluation candidates
# ---------------------------------------------------------------------------
class NegativeSampler:

    def __init__(self, num_items, num_negatives, seed, user_seen_items):
        if num_items < 2 or num_negatives < 1:
            raise ValueError("A nonempty catalog and a positive number of negatives are required.")
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)
        self.user_seen_items = user_seen_items

    def sample(self, positives: torch.Tensor, user_indices: torch.Tensor, device) -> torch.Tensor:
        """Returns candidates [B, 1 + num_negatives]; column 0 is the positive."""
        rows = []
        for positive, user in zip(positives.detach().cpu().tolist(),
                                  user_indices.detach().cpu().tolist()):
            if not 1 <= positive < self.num_items:
                raise ValueError("Positive item is outside the catalog.")
            seen = self.user_seen_items[user]
            available = self.num_items - 1 - len(seen) - int(positive not in seen)
            if available < self.num_negatives:
                raise ValueError(f"User {user} has only {available} eligible negatives.")
            if available < max(self.num_items // 2, 2 * self.num_negatives):
                pool = np.asarray([i for i in range(1, self.num_items)
                                   if i != positive and i not in seen], dtype=np.int64)
                negatives = self.rng.choice(pool, self.num_negatives, replace=False).tolist()
            else:
                negatives, selected = [], set()
                while len(negatives) < self.num_negatives:
                    proposals = self.rng.integers(1, self.num_items,
                                                  size=2 * (self.num_negatives - len(negatives)))
                    for item in proposals.tolist():
                        if item != positive and item not in seen and item not in selected:
                            negatives.append(item)
                            selected.add(item)
                            if len(negatives) == self.num_negatives:
                                break
            rows.append([positive] + negatives)
        return torch.tensor(rows, dtype=torch.long, device=device)


class FixedCandidateSampler:

    def __init__(self, candidates: torch.Tensor):
        self.candidates = candidates.cpu()

    def sample(self, positives, user_indices, device) -> torch.Tensor:
        rows = self.candidates[user_indices.detach().cpu()].to(device)
        if not torch.equal(rows[:, 0], positives):
            raise ValueError("Evaluation candidates do not match the held-out targets.")
        return rows


def prepare_evaluation_candidates(bundle: DatasetBundle, num_negatives: int, seed: int, output_dir: Path):
    tables, metadata = {}, {}
    for offset, split in ((100, "validation"), (200, "test")):
        dataset = getattr(bundle, split)
        user_positions = sorted(dataset.targets)
        if [user for user, _ in user_positions] != list(range(len(bundle.raw_user_ids))):
            raise ValueError("Evaluation requires exactly one target per user.")
        positives = torch.tensor([int(dataset.user_items[u][p]) for u, p in user_positions])
        sampler = NegativeSampler(bundle.num_items, num_negatives, seed + offset, bundle.user_seen_items)
        tables[split] = sampler.sample(positives, torch.arange(len(user_positions)), torch.device("cpu"))
        metadata[f"{split}_positions"] = np.asarray([p for _, p in user_positions])
    np.savez_compressed(
        output_dir / "evaluation_candidates.npz",
        validation=tables["validation"].numpy(), test=tables["test"].numpy(),
        raw_item_ids=np.asarray(bundle.raw_item_ids), raw_user_ids=np.asarray(bundle.raw_user_ids),
        eval_seed=np.asarray(seed), **metadata,
    )
    return FixedCandidateSampler(tables["validation"]), FixedCandidateSampler(tables["test"])


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
class UserDistinctBatchSampler(Sampler):

    def __init__(self, targets, batch_size):
        self.batch_size = batch_size
        self.groups: Dict[int, list] = {}
        for index, (user, _) in enumerate(targets):
            self.groups.setdefault(user, []).append(index)
        self.num_examples = len(targets)

    def __len__(self):
        return max(math.ceil(self.num_examples / self.batch_size),
                   max((len(v) for v in self.groups.values()), default=0))

    def __iter__(self):
        pending = {u: [idx[i] for i in torch.randperm(len(idx)).tolist()]
                   for u, idx in self.groups.items()}
        heap = [(-len(idx), torch.rand(()).item(), u) for u, idx in pending.items()]
        heapq.heapify(heap)
        batches = []
        while heap:
            selected = [heapq.heappop(heap) for _ in range(min(self.batch_size, len(heap)))]
            batch = []
            for _, _, user in selected:
                batch.append(pending[user].pop())
                if pending[user]:
                    heapq.heappush(heap, (-len(pending[user]), torch.rand(()).item(), user))
            batches.append(batch)
        for index in torch.randperm(len(batches)).tolist():
            yield batches[index]


def make_dataloader(dataset, batch_size, shuffle, num_workers, device) -> DataLoader:
    common = dict(num_workers=num_workers, pin_memory=device.type == "cuda",
                  persistent_workers=num_workers > 0)
    if shuffle:
        return DataLoader(dataset, batch_sampler=UserDistinctBatchSampler(dataset.targets, batch_size), **common)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **common)


def to_model_inputs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {"cl": batch["click_seq"], "fa": batch["favor_seq"]}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class RankingMetricAccumulator:

    def __init__(self, ks: Sequence[int] = (10, 20)):
        self.ks = tuple(ks)
        self.count = 0
        self.totals = {f"{metric}@{k}": 0.0 for k in self.ks for metric in ("HR", "NDCG")}

    def update(self, scores: torch.Tensor, positive_index: int = 0):
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite evaluation scores.")
        rank = (scores >= scores[:, positive_index:positive_index + 1]).sum(dim=1)
        gain = 1.0 / torch.log2(rank.to(torch.float64) + 1.0)
        for k in self.ks:
            hit = rank <= k
            self.totals[f"HR@{k}"] += hit.sum().item()
            self.totals[f"NDCG@{k}"] += (hit * gain).sum().item()
        self.count += scores.shape[0]

    def compute(self) -> Dict[str, float]:
        if not self.count:
            raise RuntimeError("No evaluation examples were processed.")
        return {key: value / self.count for key, value in self.totals.items()}


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------
def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(model, dataloader, optimizer, sampler, device, grad_scaler,
                    amp_enabled, grad_clip, epoch) -> float:
    model.train()
    total_loss, steps = 0.0, 0
    progress = tqdm(dataloader, desc=f"Train {epoch}")
    for cpu_batch in progress:
        batch = move_batch_to_device(cpu_batch, device)
        candidates = sampler.sample(batch["labels"], batch["user_index"], device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            loss, l_rec, l_align = model.loss(
                to_model_inputs(batch),
                pos_items=candidates[:, 0],
                neg_items=candidates[:, 1:],
                user_ids=batch["user_index"],
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")

        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item())
        steps += 1
        progress.set_postfix(loss=f"{loss.item():.4f}", rec=f"{l_rec.item():.4f}",
                             align=f"{l_align.item():.4f}")
    progress.close()
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(model, dataloader, sampler, device, ks=(10, 20)) -> Dict[str, float]:
    model.eval()
    accumulator = RankingMetricAccumulator(ks)
    for cpu_batch in tqdm(dataloader, desc="Evaluate", leave=False):
        batch = move_batch_to_device(cpu_batch, device)
        candidates = sampler.sample(batch["labels"], batch["user_index"], device)
        h = model(to_model_inputs(batch))["h"]
        # Ranking by h_u^T e_i is equivalent to ranking by sigmoid(h_u^T e_i) (Eq. 17).
        accumulator.update(model.logits(h, candidates).float(), positive_index=0)
    return accumulator.compute()


def build_model(args: argparse.Namespace, bundle: DatasetBundle) -> SBMRec:
    text_features = load_item_features(args.text_features, bundle.raw_item_ids)
    image_features = load_item_features(args.image_features, bundle.raw_item_ids)
    return SBMRec(
        num_items=bundle.num_items,
        text_features=text_features,
        image_features=image_features,
        d=args.d_model,
        seq_len=args.seq_len,
        dropout=args.dropout,
        tau=args.temperature,
        lam=args.alignment_weight,
        filter_sharing=args.filter_sharing,
        pooling=args.pooling,
        use_fourier=not args.no_fourier,
        use_wavelet=not args.no_wavelet,
        use_calibration=not args.no_calibration,
        use_transfer=not args.no_transfer,
        use_adaptive_gate=not args.no_adaptive_gate,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    bundle = build_datasets(
        data_root=args.data_root,
        seq_len=args.seq_len,
        min_user_interactions=args.min_user_interactions,
        min_favor_interactions=args.min_favor_interactions,
        min_item_freq=args.min_item_freq,
        click_behaviors=(args.click_behaviors,),
        favor_behaviors=(args.favor_behaviors,),
        use_cache=not args.no_cache,
        assume_chronological=args.assume_chronological,
    )
    print(f"[Info] train={len(bundle.train)}, validation={len(bundle.validation)}, "
          f"test={len(bundle.test)}, items={bundle.num_items - 1}")

    train_loader = make_dataloader(bundle.train, args.batch_size, True, args.num_workers, device)
    validation_loader = make_dataloader(bundle.validation, args.batch_size, False, args.num_workers, device)
    test_loader = make_dataloader(bundle.test, args.batch_size, False, args.num_workers, device)

    model = build_model(args, bundle).to(device)
    print(f"[Info] Trainable parameters: {count_trainable_parameters(model):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    amp_enabled = bool(args.amp and device.type == "cuda")
    grad_scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    train_sampler = NegativeSampler(bundle.num_items, args.num_neg, args.seed + 1, bundle.train_seen_items)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "best_sbmrec.pt"
    with (output_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    validation_sampler, test_sampler = prepare_evaluation_candidates(
        bundle, args.num_eval_neg, args.eval_seed, output_dir
    )

    early_stopping = EarlyStopping(patience=args.patience, mode="max")
    best_metrics: Dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, train_sampler, device,
                                     grad_scaler, amp_enabled, args.grad_clip, epoch)
        print(f"[Epoch {epoch}] train_loss={train_loss:.6f}")
        if epoch % args.eval_every:
            continue

        validation_metrics = evaluate(model, validation_loader, validation_sampler, device)
        print(f"[Epoch {epoch}] validation: {format_metrics(validation_metrics)}")
        monitored = validation_metrics["NDCG@20"]
        if not best_metrics or monitored > best_metrics["NDCG@20"]:
            best_metrics = dict(validation_metrics)
            save_checkpoint(path=str(best_checkpoint), model=model, optimizer=optimizer,
                            epoch=epoch, metrics=validation_metrics, args=vars(args))
        if early_stopping.step(monitored):
            print(f"[Info] Early stopping at epoch {epoch}.")
            break

    checkpoint = load_model_checkpoint(str(best_checkpoint), model, device)
    print(f"[Info] Best validation epoch {checkpoint.get('epoch')}: "
          f"{format_metrics(checkpoint.get('metrics', {}))}")
    test_metrics = evaluate(model, test_loader, test_sampler, device)
    print(f"[Test] {format_metrics(test_metrics)}")

    results = {"test": test_metrics, "alpha": model.spectral_mixing_coefficients()}
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
