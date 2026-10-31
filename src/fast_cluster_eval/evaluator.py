"""Reference and accelerated panel clustering evaluator."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

import numpy as np
import torch
from scipy import sparse

from .knn import gpu_umap_graph, gpu_knn_neighbors, normalize_log1p, panel_matrix
from .metrics import metrics


@dataclass
class EvaluationResult:
    seed: int
    labels: np.ndarray
    metrics: dict[str, float]
    backend: str
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["labels"] = self.labels.tolist()
        return d


def _leiden_from_graph(graph: sparse.csr_matrix, seed: int, resolution: float) -> np.ndarray:
    try:
        import igraph as ig
        import leidenalg
    except ImportError as e:
        raise ImportError("reference/graph Leiden requires igraph and leidenalg") from e
    coo = sparse.triu(graph, k=1).tocoo()
    g = ig.Graph(n=graph.shape[0], edges=list(zip(coo.row.tolist(), coo.col.tolist())), directed=False)
    weights = coo.data.astype(float).tolist()
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights=weights if weights else None,
        resolution_parameter=resolution,
        seed=int(seed),
    )
    return np.asarray(part.membership, dtype=np.int64)


def _scanpy_reference(x, seed: int, resolution: float, n_neighbors: int):
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(np.asarray(x, dtype=np.float64))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.neighbors(a, n_neighbors=n_neighbors, metric="euclidean", random_state=seed, use_rep="X")
    sc.tl.leiden(a, resolution=resolution, random_state=seed, flavor="leidenalg", directed=False, n_iterations=-1)
    return np.asarray(a.obs["leiden"].astype(int), dtype=np.int64), np.asarray(a.X)


def evaluate(x, y_true=None, *, seeds=(0, 1, 2), backend: Literal["reference", "cuda_exact", "cuda_graph", "cuda_proxy"] = "reference",
             n_neighbors: int = 15, resolution: float = 1.0, device: str | None = None,
             block: int = 1024, si_block: int = 1024) -> list[EvaluationResult]:
    """Evaluate one panel matrix.

    ``reference`` uses the current Scanpy/Leiden protocol. ``cuda_exact``
    keeps those reference labels and moves metric kernels to CUDA. ``cuda_graph``
    moves normalization and kNN graph construction to CUDA, then runs Leiden
    on the resulting graph; it is a fast screening path because GPU kNN can
    make different tie/neighbor choices. ``cuda_proxy`` aliases ``cuda_graph``.
    """
    import time
    arr = x.toarray() if hasattr(x, "toarray") else (x.values if hasattr(x, "values") else np.asarray(x))
    if device is None:
        device = "cuda" if backend != "reference" and torch.cuda.is_available() else "cpu"
    out = []
    for seed in seeds:
        t0 = time.perf_counter()
        if backend in ("reference", "cuda_exact"):
            pred, evaluated = _scanpy_reference(arr, seed, resolution, n_neighbors)
        else:
            z = normalize_log1p(panel_matrix(arr, device=device), target_sum=1e4)
            graph = gpu_umap_graph(z, n_neighbors=n_neighbors, block=block)
            pred = _leiden_from_graph(graph, seed, resolution)
            evaluated = z.detach().cpu().numpy()
        m = metrics(y_true, pred, evaluated, device=device if backend != "reference" else None,
                     backend="cuda" if backend != "reference" else "cpu", si_block=si_block) if y_true is not None else {}
        out.append(EvaluationResult(seed, pred, m, backend, time.perf_counter() - t0))
    return out
