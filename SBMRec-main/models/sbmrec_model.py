# -*- coding: utf-8 -*-
"""SBMRec model.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

BEHAVIORS = ("cl", "fa")
MODALITIES = ("id", "te", "im")
SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Eq. (7) and Eq. (9): single-level Haar DWT / IDWT along the temporal axis
# ---------------------------------------------------------------------------
def haar_dwt(x: torch.Tensor):
    """x: [B, N, d] with even N -> (A, D), each [B, N/2, d]."""
    even, odd = x[:, 0::2], x[:, 1::2]
    return (even + odd) / SQRT2, (even - odd) / SQRT2


def haar_idwt(a: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Inverse of haar_dwt: (A, D) each [B, N/2, d] -> [B, N, d]."""
    even, odd = (a + d) / SQRT2, (a - d) / SQRT2
    return torch.stack((even, odd), dim=2).flatten(1, 2)


# ---------------------------------------------------------------------------
# Eq. (4): behavior-modality sequence encoder
# ---------------------------------------------------------------------------
class SequenceEncoder(nn.Module):

    def __init__(self, num_items: int, d: int, seq_len: int,
                 text_features: torch.Tensor, image_features: torch.Tensor,
                 dropout: float):
        super().__init__()
        # Index 0 is padding; feature tables must be row-aligned with item ids.
        self.id_embedding = nn.Embedding(num_items, d, padding_idx=0)
        self.register_buffer("text_features", text_features.float(), persistent=False)
        self.register_buffer("image_features", image_features.float(), persistent=False)
        self.proj = nn.ModuleDict({
            "te": nn.Linear(text_features.size(1), d),
            "im": nn.Linear(image_features.size(1), d),
        })
        self.pos_embedding = nn.Embedding(seq_len, d)  # P
        self.dropout = nn.Dropout(dropout)  # NOTE: dropout placement is ours.
        nn.init.normal_(self.id_embedding.weight, std=0.02)
        with torch.no_grad():
            self.id_embedding.weight[0].zero_()

    def item_content(self, m: str, items: torch.Tensor) -> torch.Tensor:
        if m == "id":
            return self.id_embedding(items)
        table = self.text_features if m == "te" else self.image_features
        return self.proj[m](table[items])

    def forward(self, m: str, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """seq, mask: [B, N] -> H^b_{u,m}: [B, N, d]."""
        positions = torch.arange(seq.size(1), device=seq.device)
        h = self.item_content(m, seq) + self.pos_embedding(positions)
        h = self.dropout(h)
        # NOTE: padded positions are zeroed before the FFT/DWT so that padding
        # does not inject artificial spectral energy.
        return h * mask.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Section 4.1: Behavior-Modality Spectral Modulation (BMSM)
# ---------------------------------------------------------------------------
class BMSM(nn.Module):


    def __init__(self, seq_len: int, d: int, filter_sharing: str = "pair",
                 use_fourier: bool = True, use_wavelet: bool = True):
        super().__init__()
        if seq_len % 2 != 0:
            raise ValueError("N must be even for the single-level Haar DWT.")
        if filter_sharing not in {"pair", "behavior", "modality", "shared"}:
            raise ValueError(f"unknown filter_sharing: {filter_sharing}")
        self.seq_len = seq_len
        self.filter_sharing = filter_sharing
        self.use_fourier = use_fourier
        self.use_wavelet = use_wavelet
        num_freq = seq_len // 2 + 1  # K = floor(N/2) + 1

        filter_keys = sorted({self._filter_key(b, m) for b in BEHAVIORS for m in MODALITIES})
        # W_glo in R^{K x d}, W_loc in R^{(N/2) x d}. NOTE: initialized to 1
        # (identity filters); the paper does not specify the initialization.
        self.W_glo = nn.ParameterDict({k: nn.Parameter(torch.ones(num_freq, d)) for k in filter_keys})
        self.W_loc = nn.ParameterDict({k: nn.Parameter(torch.ones(seq_len // 2, d)) for k in filter_keys})
        # theta^b_m; alpha^b_m = sigmoid(theta^b_m), initialized to 0.5.
        self.theta = nn.ParameterDict({
            f"{b}_{m}": nn.Parameter(torch.zeros(())) for b in BEHAVIORS for m in MODALITIES
        })

    def _filter_key(self, b: str, m: str) -> str:
        return {"pair": f"{b}_{m}", "behavior": b, "modality": m,
                "shared": "all"}[self.filter_sharing]

    def fourier(self, h: torch.Tensor, b: str, m: str) -> torch.Tensor:
        """Eqs. (5)-(6): G = IFFT(FFT(H) * W_glo) with output length N."""
        spec = torch.fft.rfft(h, dim=1, norm="ortho")              # [B, K, d], complex
        spec = spec * self.W_glo[self._filter_key(b, m)]           # real filter, per (k, dim)
        return torch.fft.irfft(spec, n=self.seq_len, dim=1, norm="ortho")

    def wavelet(self, h: torch.Tensor, b: str, m: str) -> torch.Tensor:
        """Eqs. (7)-(9): keep A, modulate D, reconstruct by IDWT."""
        a, d = haar_dwt(h)
        d = d * self.W_loc[self._filter_key(b, m)]
        return haar_idwt(a, d)

    def alpha(self, b: str, m: str) -> torch.Tensor:
        return torch.sigmoid(self.theta[f"{b}_{m}"])

    def forward(self, h: torch.Tensor, b: str, m: str) -> torch.Tensor:
        """Eq. (10). Ablations of Section 5.3 use the remaining branch directly:
        w/o BMSM-F -> Z = L ; w/o BMSM-W -> Z = G ; both off (Table 5 "No") -> Z = H.
        """
        if self.use_fourier and self.use_wavelet:
            a = self.alpha(b, m)
            return a * self.fourier(h, b, m) + (1.0 - a) * self.wavelet(h, b, m)
        if self.use_wavelet:
            return self.wavelet(h, b, m)
        if self.use_fourier:
            return self.fourier(h, b, m)
        return h



class SequencePooling(nn.Module):


    def __init__(self, d: int, mode: str = "attention"):
        super().__init__()
        if mode not in {"attention", "mean", "last"}:
            raise ValueError(f"unknown pooling mode: {mode}")
        self.mode = mode
        self.proj = nn.Linear(d, d)
        self.score = nn.Linear(d, 1, bias=False)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """z: [B, N, d], mask: [B, N] (True = valid) -> [B, d]."""
        maskf = mask.unsqueeze(-1).to(z.dtype)
        has_any = mask.any(dim=1, keepdim=True).to(z.dtype)  # NOTE: empty history -> zero vector
        if self.mode == "mean":
            return (z * maskf).sum(1) / maskf.sum(1).clamp_min(1.0)
        if self.mode == "last":
            pos = torch.arange(1, z.size(1) + 1, device=z.device)
            last = (mask.long() * pos).argmax(dim=1)
            return z[torch.arange(z.size(0), device=z.device), last] * has_any
        scores = self.score(torch.tanh(self.proj(z))).squeeze(-1)          # [B, N]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        beta = torch.softmax(scores, dim=1) * mask.to(z.dtype)            # 0 at padding
        return (beta.unsqueeze(-1) * z).sum(1) * has_any


# ---------------------------------------------------------------------------
# Section 4.2: Behavior-Modality Calibration and Transfer (BMCT)
# ---------------------------------------------------------------------------
class BMCT(nn.Module):


    def __init__(self, d: int, use_calibration: bool = True, use_adaptive_gate: bool = True):
        super().__init__()
        self.use_calibration = use_calibration
        self.use_adaptive_gate = use_adaptive_gate
        # a_b in R^d: behavior-specific query, shared across users.
        self.query = nn.ParameterDict({b: nn.Parameter(torch.randn(d) * 0.02) for b in BEHAVIORS})
        self.W_sh = nn.Linear(d, d, bias=False)   # shared by both behaviors and all modalities
        self.gate = nn.Linear(2 * d, 1)           # w_g in R^{2d}, b_g in R; shared across modalities

    def forward(self, z: Dict[str, Dict[str, torch.Tensor]]):
        """z[b][m]: [B, d] -> dict with z_tilde, r, z_hat, omega, g."""
        z_tilde, r, omega = {}, {}, {}
        for b in BEHAVIORS:
            stacked = torch.stack([z[b][m] for m in MODALITIES], dim=1)     # [B, |M|, d]
            if self.use_calibration:
                w = torch.softmax(stacked @ self.query[b], dim=1)            # Eq. (12), [B, |M|]
            else:
                w = stacked.new_full(stacked.shape[:2], 1.0 / len(MODALITIES))
            omega[b] = w
            z_tilde[b] = {m: w[:, i:i + 1] * stacked[:, i] for i, m in enumerate(MODALITIES)}
            r[b] = {m: self.W_sh(z_tilde[b][m]) for m in MODALITIES}         # Eq. (13)

        z_hat, g = {}, {}
        for m in MODALITIES:
            r_cl, r_fa = r["cl"][m], r["fa"][m]
            if self.use_adaptive_gate:
                g[m] = torch.sigmoid(self.gate(torch.cat([r_cl, r_fa], dim=-1)))  # Eq. (14), [B, 1]
            else:
                g[m] = r_cl.new_full((r_cl.size(0), 1), 0.5)
            z_hat[m] = g[m] * r_cl + (1.0 - g[m]) * r_fa                      # Eq. (15)
        return {"z_tilde": z_tilde, "r": r, "z_hat": z_hat, "omega": omega, "g": g}

class SBMRec(nn.Module):

    def __init__(self, num_items: int, text_features: torch.Tensor,
                 image_features: torch.Tensor, d: int = 256, seq_len: int = 50,
                 dropout: float = 0.1, tau: float = 0.1, lam: float = 0.1,
                 filter_sharing: str = "pair", pooling: str = "attention",
                 use_fourier: bool = True, use_wavelet: bool = True,
                 use_calibration: bool = True, use_transfer: bool = True,
                 use_adaptive_gate: bool = True):
        super().__init__()
        self.tau = tau
        self.lam = lam
        self.use_transfer = use_transfer
        self.encoder = SequenceEncoder(num_items, d, seq_len, text_features,
                                       image_features, dropout)
        self.bmsm = BMSM(seq_len, d, filter_sharing, use_fourier, use_wavelet)
        self.pooling = SequencePooling(d, pooling)
        self.bmct = BMCT(d, use_calibration, use_adaptive_gate)
        in_dim = (3 if use_transfer else 2) * d
        # NOTE: the paper only says "an MLP"; this is a two-layer MLP.
        self.mlp = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d, d))

    # ---- representation -------------------------------------------------
    def forward(self, seqs: Dict[str, torch.Tensor]) -> Dict[str, object]:
        z = {}
        for b in BEHAVIORS:
            mask = seqs[b] != 0                                     # V_{u,b}
            z[b] = {}
            for m in MODALITIES:
                h = self.encoder(m, seqs[b], mask)                  # Eq. (4)
                zz = self.bmsm(h, b, m)                             # Eqs. (5)-(10)
                z[b][m] = self.pooling(zz, mask)                    # Eq. (11)
        out = self.bmct(z)                                          # Eqs. (12)-(15)

        terms = [sum(out["z_tilde"]["cl"][m] for m in MODALITIES),
                 sum(out["z_tilde"]["fa"][m] for m in MODALITIES)]
        if self.use_transfer:
            terms.append(sum(out["z_hat"][m] for m in MODALITIES))
        out["h"] = self.mlp(torch.cat(terms, dim=-1))               # Eq. (16)
        return out

    # ---- Eq. (17) ---------------------------------------------------------
    def logits(self, h: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        e = self.encoder.id_embedding(items)
        return (h.unsqueeze(1) * e).sum(-1)

    @torch.no_grad()
    def predict(self, seqs: Dict[str, torch.Tensor], candidates: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(self.forward(seqs)["h"], candidates))

    # ---- Eq. (18) ---------------------------------------------------------
    @staticmethod
    def rec_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
        """BCE averaged over all positive and sampled negative pairs in D."""
        logits = torch.cat([pos_logits.reshape(-1), neg_logits.reshape(-1)])
        labels = torch.cat([torch.ones_like(pos_logits.reshape(-1)),
                            torch.zeros_like(neg_logits.reshape(-1))])
        return F.binary_cross_entropy_with_logits(logits, labels)

    # ---- Eq. (19) ---------------------------------------------------------
    def align_loss(self, r: Dict[str, Dict[str, torch.Tensor]],
                   user_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        losses = []
        for m in MODALITIES:
            rc = F.normalize(r["cl"][m], dim=-1)
            rf = F.normalize(r["fa"][m], dim=-1)
            sim = rc @ rf.t() / self.tau                            # cosine / tau, [B, B]
            if user_ids is not None:
                same = user_ids.unsqueeze(0) == user_ids.unsqueeze(1)
                same.fill_diagonal_(False)
                sim = sim.masked_fill(same, torch.finfo(sim.dtype).min)
            target = torch.arange(sim.size(0), device=sim.device)
            losses.append(F.cross_entropy(sim, target))             # mean over users in U_B
        return torch.stack(losses).mean()                           # mean over modalities

    # ---- Eq. (20) ---------------------------------------------------------
    def loss(self, seqs: Dict[str, torch.Tensor], pos_items: torch.Tensor,
             neg_items: torch.Tensor, user_ids: Optional[torch.Tensor] = None):
        out = self.forward(seqs)
        pos = self.logits(out["h"], pos_items.unsqueeze(1)).squeeze(1)
        neg = self.logits(out["h"], neg_items)
        l_rec = self.rec_loss(pos, neg)
        l_align = self.align_loss(out["r"], user_ids)
        return l_rec + self.lam * l_align, l_rec, l_align

    # ---- analysis helpers -------------------------------------------------
    def spectral_mixing_coefficients(self) -> Dict[str, float]:
        return {f"{b}_{m}": self.bmsm.alpha(b, m).item() for b in BEHAVIORS for m in MODALITIES}
