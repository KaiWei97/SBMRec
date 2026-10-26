# -*- coding: utf-8 -*-
"""Reusable neural modules for SBMRec.

The implementation follows the modules in the SBMRec paper:

1. Modality encoders for ID, text, and image features.
2. Behavior-modality spectral decomposition with FFT/IFFT.
3. Weak-to-strong spectral enhancement.
4. Strong-guided weak-behavior denoising.
5. Masked attention pooling and user-adaptive fusion.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _read_numpy_matrix(path: str) -> np.ndarray:
    """Load a two-dimensional feature matrix from ``.npy`` or ``.npz``."""
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        keys = list(loaded.keys())
        if not keys:
            raise ValueError(f"No array was found in {path}.")
        matrix = loaded[keys[0]]
        loaded.close()
    else:
        matrix = loaded
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2-D feature matrix at {path}, but received shape {matrix.shape}."
        )
    return matrix.astype(np.float32, copy=False)


def _align_feature_matrix(
    matrix: np.ndarray,
    num_items: int,
    raw_item_ids: Optional[Sequence[str]],
    feature_name: str,
    strict: bool,
) -> np.ndarray:
    """Align an external item-feature matrix to the model's contiguous item IDs.

    Alignment is attempted in this order:

    1. Matrix already contains ``num_items`` rows, including padding row 0.
    2. Matrix contains ``num_items - 1`` rows; a zero padding row is prepended.
    3. Raw item IDs can be interpreted as integer row indices into the matrix.
    4. Non-strict fallback: copy the overlapping prefix and zero-fill the rest.
    """
    rows, dim = matrix.shape
    if rows == num_items:
        aligned = matrix.copy()
        aligned[0] = 0.0
        return aligned

    if rows == num_items - 1:
        aligned = np.zeros((num_items, dim), dtype=np.float32)
        aligned[1:] = matrix
        return aligned

    if raw_item_ids is not None and len(raw_item_ids) == num_items:
        raw_indices: List[int] = []
        can_index = True
        for raw_id in raw_item_ids[1:]:
            try:
                raw_index = int(str(raw_id))
            except (TypeError, ValueError):
                can_index = False
                break
            if raw_index < 0 or raw_index >= rows:
                can_index = False
                break
            raw_indices.append(raw_index)
        if can_index:
            aligned = np.zeros((num_items, dim), dtype=np.float32)
            aligned[1:] = matrix[np.asarray(raw_indices, dtype=np.int64)]
            return aligned

    message = (
        f"Cannot exactly align {feature_name} features with shape {matrix.shape} "
        f"to num_items={num_items}."
    )
    if strict:
        raise ValueError(
            message
            + " Supply a matrix aligned to remapped item IDs, or use integer raw item IDs "
            "that index the feature rows."
        )

    warnings.warn(
        message + " Falling back to prefix copy with zero padding/truncation.",
        RuntimeWarning,
    )
    aligned = np.zeros((num_items, dim), dtype=np.float32)
    copy_rows = min(rows, num_items)
    aligned[:copy_rows] = matrix[:copy_rows]
    aligned[0] = 0.0
    return aligned


class ContentEmbedding(nn.Module):
    """Item-content lookup followed by a projection to the shared latent space."""

    def __init__(
        self,
        num_items: int,
        d_model: int,
        feature_path: Optional[str],
        raw_item_ids: Optional[Sequence[str]],
        freeze: bool,
        strict_alignment: bool,
        feature_name: str,
    ) -> None:
        super().__init__()
        self.feature_name = feature_name
        self.uses_pretrained_features = bool(feature_path)

        if feature_path:
            if not os.path.isfile(feature_path):
                raise FileNotFoundError(
                    f"{feature_name} feature file does not exist: {feature_path}"
                )
            matrix = _read_numpy_matrix(feature_path)
            matrix = _align_feature_matrix(
                matrix=matrix,
                num_items=num_items,
                raw_item_ids=raw_item_ids,
                feature_name=feature_name,
                strict=strict_alignment,
            )
            weights = torch.from_numpy(matrix)
            self.lookup = nn.Embedding.from_pretrained(
                weights,
                freeze=freeze,
                padding_idx=0,
            )
            input_dim = int(matrix.shape[1])
        else:
            self.lookup = nn.Embedding(num_items, d_model, padding_idx=0)
            nn.init.normal_(self.lookup.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.lookup.weight[0].zero_()
            input_dim = d_model

        self.projection = (
            nn.Identity() if input_dim == d_model else nn.Linear(input_dim, d_model)
        )

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.lookup(item_ids))


class MultiModalItemEncoder(nn.Module):
    """Encode ID, text, and image item sequences into a common dimension."""

    modalities: Tuple[str, ...] = ("id", "text", "image")

    def __init__(
        self,
        num_items: int,
        d_model: int,
        text_embeddings_path: Optional[str] = None,
        image_embeddings_path: Optional[str] = None,
        raw_item_ids: Optional[Sequence[str]] = None,
        freeze_content: bool = True,
        strict_feature_alignment: bool = True,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model

        self.id_embedding = nn.Embedding(num_items, d_model, padding_idx=0)
        nn.init.normal_(self.id_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.id_embedding.weight[0].zero_()

        self.text_embedding = ContentEmbedding(
            num_items=num_items,
            d_model=d_model,
            feature_path=text_embeddings_path,
            raw_item_ids=raw_item_ids,
            freeze=freeze_content,
            strict_alignment=strict_feature_alignment,
            feature_name="text",
        )
        self.image_embedding = ContentEmbedding(
            num_items=num_items,
            d_model=d_model,
            feature_path=image_embeddings_path,
            raw_item_ids=raw_item_ids,
            freeze=freeze_content,
            strict_alignment=strict_feature_alignment,
            feature_name="image",
        )

    def encode(self, modality: str, item_ids: torch.Tensor) -> torch.Tensor:
        if modality == "id":
            return self.id_embedding(item_ids)
        if modality == "text":
            return self.text_embedding(item_ids)
        if modality == "image":
            return self.image_embedding(item_ids)
        raise KeyError(f"Unsupported modality: {modality}")

    def target_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Return the item embedding used by the prediction layer."""
        return self.id_embedding(item_ids)


class BehaviorModalitySpectralDecomposition(nn.Module):
    """Split a behavior-modality sequence into low/high frequency branches.

    ``torch.fft.rfft`` keeps the non-redundant spectrum of a real-valued
    sequence. The cutoff is ``floor(K / gamma)`` where ``K = N // 2 + 1``.
    Each branch is reconstructed by zero-masking the complementary frequency
    coefficients and applying ``irfft`` with the original sequence length.
    """

    def __init__(
        self,
        max_seq_len: int,
        d_model: int,
        gamma: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.gamma = float(gamma)
        self.position_embedding = nn.Parameter(torch.empty(max_seq_len, d_model))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def prepare_time_domain(
        self,
        sequence: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Add positional information and zero out padded positions."""
        if sequence.ndim != 3:
            raise ValueError(
                f"Expected sequence [B, N, D], but received {tuple(sequence.shape)}."
            )
        batch_size, seq_len, d_model = sequence.shape
        del batch_size
        if d_model != self.d_model:
            raise ValueError(
                f"Expected d_model={self.d_model}, but received {d_model}."
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )
        if valid_mask.shape != sequence.shape[:2]:
            raise ValueError(
                f"Mask shape {tuple(valid_mask.shape)} does not match sequence shape "
                f"{tuple(sequence.shape[:2])}."
            )
        position = self.position_embedding[:seq_len].unsqueeze(0)
        prepared = self.dropout(sequence + position)
        return prepared * valid_mask.unsqueeze(-1).to(prepared.dtype)

    def forward(
        self,
        sequence: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prepared = self.prepare_time_domain(sequence, valid_mask)
        seq_len = prepared.size(1)

        # FFT in float32 is more stable and also avoids fp16 FFT restrictions on
        # non-power-of-two sequence lengths under mixed precision.
        original_dtype = prepared.dtype
        spectrum = torch.fft.rfft(prepared.float(), dim=1, norm="ortho")
        num_frequencies = spectrum.size(1)
        cutoff = int(math.floor(num_frequencies / self.gamma))
        cutoff = max(1, min(num_frequencies, cutoff))

        low_spectrum = torch.zeros_like(spectrum)
        low_spectrum[:, :cutoff, :] = spectrum[:, :cutoff, :]
        high_spectrum = spectrum - low_spectrum

        low = torch.fft.irfft(
            low_spectrum, n=seq_len, dim=1, norm="ortho"
        ).to(original_dtype)
        high = torch.fft.irfft(
            high_spectrum, n=seq_len, dim=1, norm="ortho"
        ).to(original_dtype)

        mask = valid_mask.unsqueeze(-1).to(original_dtype)
        return low * mask, high * mask


def masked_softmax(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    dim: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Numerically stable softmax that returns zero on invalid positions."""
    mask = valid_mask.to(dtype=torch.bool)
    while mask.ndim < logits.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(logits)

    min_value = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask, min_value)
    probabilities = torch.softmax(masked_logits, dim=dim)
    probabilities = probabilities * mask.to(probabilities.dtype)
    normalizer = probabilities.sum(dim=dim, keepdim=True).clamp_min(eps)
    return probabilities / normalizer


class BehaviorModalitySpectralRefinement(nn.Module):
    """Two-stage cross-behavior refinement from the SBMRec paper."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.context_projection = nn.Linear(d_model, d_model, bias=False)
        self.strong_layer_norm = nn.LayerNorm(d_model)
        self.weak_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        weak_features: torch.Tensor,
        strong_features: torch.Tensor,
        weak_mask: torch.Tensor,
        strong_mask: torch.Tensor,
        enable_weak_to_strong: bool = True,
        enable_strong_guided_denoising: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if weak_features.size(-1) != self.d_model:
            raise ValueError("Weak feature dimension does not match d_model.")
        if strong_features.size(-1) != self.d_model:
            raise ValueError("Strong feature dimension does not match d_model.")

        queries = self.query_projection(strong_features)
        keys = self.key_projection(weak_features)
        relation = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(
            self.d_model
        )

        weak_to_strong_attention = masked_softmax(
            relation,
            weak_mask.unsqueeze(1),
            dim=-1,
        )
        weak_context = torch.matmul(
            weak_to_strong_attention,
            self.value_projection(weak_features),
        )
        if enable_weak_to_strong:
            enhanced_strong = self.strong_layer_norm(
                strong_features + self.dropout(weak_context)
            )
        else:
            enhanced_strong = strong_features
        enhanced_strong = enhanced_strong * strong_mask.unsqueeze(-1).to(
            enhanced_strong.dtype
        )

        strong_to_weak_attention = masked_softmax(
            relation.transpose(1, 2),
            strong_mask.unsqueeze(1),
            dim=-1,
        )
        strong_context = torch.matmul(
            strong_to_weak_attention,
            self.context_projection(enhanced_strong),
        )
        gate = torch.sigmoid(
            self.weak_gate(torch.cat([weak_features, strong_context], dim=-1))
        )
        if enable_strong_guided_denoising:
            denoised_weak = gate * weak_features
        else:
            denoised_weak = weak_features
        denoised_weak = denoised_weak * weak_mask.unsqueeze(-1).to(
            denoised_weak.dtype
        )

        auxiliary = {
            "relation": relation,
            "weak_to_strong_attention": weak_to_strong_attention,
            "strong_to_weak_attention": strong_to_weak_attention,
            "weak_retention_gate": gate,
        }
        return denoised_weak, enhanced_strong, auxiliary


class MaskedAttentionPooling(nn.Module):
    """Attention pooling over valid behavior-modality-frequency tokens."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1, bias=False),
        )

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.scorer(features).squeeze(-1)
        weights = masked_softmax(scores, valid_mask, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * features, dim=1)
        return pooled, weights


class UserAdaptiveFusion(nn.Module):
    """Fuse weak and strong behavior representations with a user-wise weight."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        weak_representation: torch.Tensor,
        strong_representation: torch.Tensor,
        mode: str = "adaptive",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mode == "adaptive":
            weight = torch.sigmoid(
                self.fusion_mlp(
                    torch.cat(
                        [weak_representation, strong_representation], dim=-1
                    )
                )
            )
        elif mode == "average":
            weight = torch.full(
                (weak_representation.size(0), 1),
                fill_value=0.5,
                dtype=weak_representation.dtype,
                device=weak_representation.device,
            )
        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")

        fused = weight * weak_representation + (1.0 - weight) * strong_representation
        return fused, weight


def ensure_nonempty_sequence(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    fallback_token: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Insert a learned fallback token for samples with an empty behavior history."""
    empty = ~valid_mask.any(dim=1)
    if not torch.any(empty):
        return features, valid_mask

    batch_size, _, d_model = features.shape
    replacement = fallback_token.view(1, 1, d_model).expand(batch_size, 1, d_model)
    first_token = torch.where(
        empty.view(batch_size, 1, 1), replacement, features[:, :1, :]
    )
    features = torch.cat([first_token, features[:, 1:, :]], dim=1)

    first_valid = torch.where(
        empty.view(batch_size, 1),
        torch.ones((batch_size, 1), dtype=torch.bool, device=valid_mask.device),
        valid_mask[:, :1],
    )
    valid_mask = torch.cat([first_valid, valid_mask[:, 1:]], dim=1)
    return features, valid_mask
