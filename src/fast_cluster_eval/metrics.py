"""Exact or numerically controlled clustering metrics on CPU and CUDA."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def _labels(x, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(np.asarray(x), device=device, dtype=torch.long).reshape(-1)


def _contingency(y_true, y_pred, device=None) -> tuple[torch.Tensor, int, int]:
    a = _labels(y_true, device)
    b = _labels(y_pred, device)
    if a.numel() != b.numel():
        raise ValueError("y_true and y_pred must have the same length")
    _, ai = torch.unique(a, sorted=True, return_inverse=True)
    _, bi = torch.unique(b, sorted=True, return_inverse=True)
    na = int(ai.max().item()) + 1 if ai.numel() else 0
    nb = int(bi.max().item()) + 1 if bi.numel() else 0
    c = torch.bincount(ai * max(nb, 1) + bi, minlength=max(na * nb, 1))
    return c.reshape(na, nb), na, nb


def normalized_mutual_information(y_true, y_pred, device: str | torch.device | None = None,
                                  average_method: Literal["arithmetic"] = "arithmetic") -> float:
    """sklearn-compatible arithmetic-mean NMI using a contingency matrix."""
    if average_method != "arithmetic":
        raise ValueError("only arithmetic normalization is supported")
    c, _, _ = _contingency(y_true, y_pred, device)
    if c.numel() == 0 or c.sum() == 0:
        return 0.0
    z = c.to(torch.float64)
    n = z.sum()
    rows = z.sum(1, keepdim=True)
    cols = z.sum(0, keepdim=True)
    nz = z > 0
    mi = torch.where(nz, z / n * torch.log((z * n) / (rows * cols).clamp_min(1e-300)), torch.zeros_like(z)).sum()
    hr = -(rows / n).clamp_min(1e-300).log().mul(rows / n).sum()
    hc = -(cols / n).clamp_min(1e-300).log().mul(cols / n).sum()
    den = hr + hc
    return float((2 * mi / den).item()) if den.item() > 0 else 1.0


def adjusted_rand(y_true, y_pred, device: str | torch.device | None = None) -> float:
    """sklearn-compatible adjusted Rand index from GPU contingency counts."""
    c, _, _ = _contingency(y_true, y_pred, device)
    n = int(c.sum().item())
    if n < 2:
        return 1.0
    z = c.to(torch.float64)
    comb = lambda x: x * (x - 1) / 2
    nij = comb(z).sum()
    ai = comb(z.sum(1)).sum()
    bj = comb(z.sum(0)).sum()
    total = torch.tensor(float(comb(torch.tensor(float(n), device=z.device))), dtype=torch.float64, device=z.device)
    expected = ai * bj / total
    max_index = (ai + bj) / 2
    den = max_index - expected
    return float(((nij - expected) / den).item()) if den.item() else 1.0


def _silhouette_blockwise(x: torch.Tensor, labels: torch.Tensor, block: int = 1024) -> float:
    """Exact Euclidean silhouette without materializing an n-by-n matrix."""
    n = x.shape[0]
    unique = torch.unique(labels, sorted=True)
    if unique.numel() < 2 or unique.numel() >= n:
        return float("nan")
    sums = torch.zeros((n, unique.numel()), dtype=torch.float64, device=x.device)
    counts = torch.bincount(labels, minlength=int(unique.max().item()) + 1).to(torch.float64)
    for start in range(0, n, block):
        stop = min(n, start + block)
        d = torch.cdist(x[start:stop].to(torch.float64), x.to(torch.float64), p=2)
        for j, cls in enumerate(unique):
            sums[start:stop, j] = d[:, labels == cls].sum(1)
    own = labels.unsqueeze(1) == unique.unsqueeze(0)
    mean = sums / counts[unique].unsqueeze(0).clamp_min(1)
    # Exclude self-distance from the within-cluster mean.
    own_idx = torch.searchsorted(unique, labels)
    mean[torch.arange(n, device=x.device), own_idx] *= counts[labels] / (counts[labels] - 1).clamp_min(1)
    a = mean[torch.arange(n, device=x.device), own_idx]
    mean = mean.masked_fill(own, float("inf"))
    b = mean.min(1).values
    return float(((b - a) / torch.maximum(a, b).clamp_min(1e-300)).mean().item())


def silhouette(x, labels, *, device: str | torch.device | None = None, block: int = 1024,
               backend: Literal["auto", "cpu", "cuda"] = "auto") -> float:
    arr = np.asarray(x) if not isinstance(x, torch.Tensor) else x
    if backend == "cpu" or (backend == "auto" and device in (None, "cpu")):
        y = np.asarray(labels)
        if len(np.unique(y)) < 2 or len(np.unique(y)) >= len(y):
            return float("nan")
        return float(silhouette_score(np.asarray(arr), y, metric="euclidean"))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return _silhouette_blockwise(torch.as_tensor(arr, device=device), _labels(labels, device), block)


def metrics(y_true, y_pred, x=None, *, device=None, backend="auto", si_block=1024) -> dict[str, float]:
    use_gpu = backend in ("cuda", "auto") and device not in (None, "cpu")
    d = device if use_gpu else None
    out = {
        "NMI": normalized_mutual_information(y_true, y_pred, d),
        "ARI": adjusted_rand(y_true, y_pred, d),
    }
    out["SI"] = silhouette(x, y_pred, device=d, block=si_block, backend="cuda" if use_gpu else "cpu") if x is not None else float("nan")
    return out


def sklearn_metrics(y_true, y_pred, x=None) -> dict[str, float]:
    out = {
        "NMI": float(normalized_mutual_info_score(np.asarray(y_true), np.asarray(y_pred))),
        "ARI": float(adjusted_rand_score(np.asarray(y_true), np.asarray(y_pred))),
    }
    out["SI"] = float(silhouette_score(np.asarray(x), np.asarray(y_pred))) if x is not None and 1 < len(set(y_pred)) < len(y_pred) else float("nan")
    return out
