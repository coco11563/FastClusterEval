"""Memory-bounded kNN graph construction."""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def panel_matrix(x, *, device=None, dtype=torch.float32) -> torch.Tensor:
    if hasattr(x, "toarray"):
        x = x.toarray()
    if hasattr(x, "values"):
        x = x.values
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def normalize_log1p(x: torch.Tensor, target_sum: float = 1e4) -> torch.Tensor:
    x = x.to(torch.float32)
    scale = target_sum / x.sum(1, keepdim=True).clamp_min(1e-12)
    return torch.log1p(x * scale)


def gpu_knn(x: torch.Tensor, n_neighbors: int = 15, *, block: int = 1024,
            include_self: bool = False) -> sparse.csr_matrix:
    """Return a symmetric kNN adjacency, using bounded GPU matmul/topk."""
    if x.device.type != "cuda":
        raise ValueError("gpu_knn requires a CUDA tensor")
    n = x.shape[0]
    k = n_neighbors + (1 if include_self else 0)
    x = x.contiguous()
    norms = (x * x).sum(1).reshape(1, -1)
    rows, cols = [], []
    for start in range(0, n, block):
        stop = min(n, start + block)
        dist = norms + (x[start:stop] * x[start:stop]).sum(1, keepdim=True) - 2 * (x[start:stop] @ x.T)
        if not include_self:
            dist[:, start:stop].fill_(float("inf"))
        idx = torch.topk(dist, k=min(k, n - (0 if include_self else 1)), largest=False, dim=1).indices
        rr = torch.arange(start, stop, device=x.device).unsqueeze(1).expand_as(idx)
        rows.append(rr.reshape(-1).cpu().numpy())
        cols.append(idx.reshape(-1).cpu().numpy())
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    a = sparse.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(n, n))
    return a.maximum(a.T)


def gpu_knn_neighbors(x: torch.Tensor, n_neighbors: int = 15, *, block: int = 1024,
                      include_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return only kNN indices and distances, minimizing the CUDA→CPU transfer.

    The returned arrays are suitable for Scanpy's UMAP fuzzy connectivity
    routine. No n-by-n adjacency is materialized on the GPU or transferred.
    """
    if x.device.type != "cuda":
        raise ValueError("gpu_knn_neighbors requires a CUDA tensor")
    n = x.shape[0]
    k = n_neighbors + (1 if include_self else 0)
    idx, vals = gpu_knn_neighbors_device(x, n_neighbors, block=block, include_self=include_self)
    return idx.cpu().numpy(), vals.cpu().numpy()


def gpu_knn_neighbors_device(x: torch.Tensor, n_neighbors: int = 15, *, block: int = 1024,
                             include_self: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-resident counterpart of :func:`gpu_knn_neighbors`."""
    if x.device.type != "cuda":
        raise ValueError("gpu_knn_neighbors_device requires a CUDA tensor")
    n = x.shape[0]
    k = n_neighbors + (1 if include_self else 0)
    x = x.contiguous()
    norms = (x * x).sum(1).reshape(1, -1)
    all_i, all_d = [], []
    for start in range(0, n, block):
        stop = min(n, start + block)
        dist = norms + (x[start:stop] * x[start:stop]).sum(1, keepdim=True) - 2 * (x[start:stop] @ x.T)
        if not include_self:
            dist[:, start:stop].fill_(float("inf"))
        vals, idx = torch.topk(dist, k=min(k, n - (0 if include_self else 1)), largest=False, dim=1)
        all_i.append(idx)
        all_d.append(vals.clamp_min(0).sqrt())
    return torch.cat(all_i, dim=0), torch.cat(all_d, dim=0)


def gpu_umap_graph(x: torch.Tensor, n_neighbors: int = 15, *, block: int = 1024,
                   search_steps: int = 32) -> sparse.csr_matrix:
    """GPU fuzzy-connectivity graph with only an edge-list transfer to CPU.

    This is a self-contained approximation of UMAP's smooth kNN weights. The
    expensive distance search and per-row sigma solve stay on CUDA. Only
    ``n_cells * k`` indices and weights are copied back for Leiden's CPU graph
    object, rather than an n-by-n matrix.
    """
    idx, dist = gpu_knn_neighbors_device(x, n_neighbors, block=block)
    rho = dist[:, :1]
    target = float(np.log2(max(n_neighbors, 2)))
    lo = torch.full_like(rho, 1e-4)
    hi = dist[:, -1:].clamp_min(1e-3)
    for _ in range(search_steps):
        mid = (lo + hi) * 0.5
        val = torch.exp(-(dist - rho).clamp_min(0) / mid).sum(1, keepdim=True)
        hi = torch.where(val > target, mid, hi)
        lo = torch.where(val > target, lo, mid)
    sigma = ((lo + hi) * 0.5).clamp_min(1e-6)
    weight = torch.exp(-(dist - rho).clamp_min(0) / sigma)
    r = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(idx)
    r = r.reshape(-1).cpu().numpy()
    c = idx.reshape(-1).cpu().numpy()
    w = weight.reshape(-1).cpu().numpy().astype(np.float32)
    a = sparse.coo_matrix((w, (r, c)), shape=(x.shape[0], x.shape[0])).tocsr()
    return a.maximum(a.T)


def cpu_knn(x, n_neighbors: int = 15) -> sparse.csr_matrix:
    arr = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean", algorithm="auto").fit(arr)
    graph = nn.kneighbors_graph(arr, mode="connectivity")
    graph.setdiag(0)
    return graph.maximum(graph.T).tocsr()
