"""Atomic multi-dataset evaluation with stable result serialization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Iterable
import time
import math

import numpy as np

from .evaluator import evaluate
from .io import DatasetInput, SCHEMA_VERSION


def _json_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _one(req: DatasetInput, *, backend, device, block, si_block, cugraph_resolution) -> dict:
    started = time.perf_counter()
    try:
        req.validate()
        matrix = req.matrix
        truth = req.truth_labels
        if req.row_indices is not None:
            rows = np.asarray(req.row_indices, dtype=np.int64)
            matrix = matrix[rows]
            truth = np.asarray(truth)[rows]
        panels = []
        for panel_id, indices in req.panels().items():
            result = evaluate(
                matrix[:, indices], truth,
                seeds=req.seeds, backend=backend, device=device,
                block=block, si_block=si_block, cugraph_resolution=cugraph_resolution,
            )
            rows = []
            for r in result:
                rows.append({
                    "seed": r.seed,
                    "metrics": {k: _json_number(v) for k, v in r.metrics.items() if k in req.metrics},
                    "seconds": float(r.seconds),
                    "backend": r.backend,
                })
            aggregate = {}
            for k in req.metrics:
                vals = [row["metrics"].get(k) for row in rows if row["metrics"].get(k) is not None]
                aggregate[k] = float(np.mean(vals)) if vals else None
            panels.append({
                "panel_id": panel_id,
                "selected_matrix_indices": indices,
                "metrics": aggregate,
                "seeds": rows,
            })
        return {
            "dataset_id": req.dataset_id,
            "status": "ok",
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "panels": panels,
            "metadata": req.metadata,
            "seconds": float(time.perf_counter() - started),
        }
    except Exception as exc:  # each dataset is an atomic result unit
        return {
            "dataset_id": req.dataset_id,
            "status": "error",
            "shape": list(getattr(req.matrix, "shape", ())),
            "panels": [],
            "metadata": req.metadata,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "seconds": float(time.perf_counter() - started),
        }


def evaluate_batch(requests: Iterable[DatasetInput], *, backend="reference", device=None,
                   max_concurrency: int = 1, block: int = 1024, si_block: int = 1024,
                   cugraph_resolution: float | None = None) -> dict:
    """Evaluate many datasets concurrently and commit one result per dataset.

    CUDA work is safe to parallelize at the dataset boundary, but callers
    should set max_concurrency from available VRAM. A failed dataset produces
    an error object and does not discard successful datasets.
    """
    reqs = list(requests)
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    results = [None] * len(reqs)
    if max_concurrency == 1:
        for i, req in enumerate(reqs):
            results[i] = _one(req, backend=backend, device=device, block=block, si_block=si_block,
                              cugraph_resolution=cugraph_resolution)
    else:
        with ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="fce") as pool:
            futures = {
                pool.submit(_one, req, backend=backend, device=device, block=block, si_block=si_block,
                            cugraph_resolution=cugraph_resolution): i
                for i, req in enumerate(reqs)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "datasets": results,
        "summary": {
            "count": len(results),
            "ok": sum(r["status"] == "ok" for r in results),
            "error": sum(r["status"] == "error" for r in results),
        },
    }
