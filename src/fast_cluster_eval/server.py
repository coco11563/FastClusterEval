"""Resident HTTP service for GPU cluster evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

import numpy as np

from .batch import evaluate_batch
from .evaluator import evaluate
from .io import DatasetInput, request_from_dict
from .metrics import metrics
from .registry import DatasetRegistry


def _json_number(value):
    value = float(value)
    return value if value == value and abs(value) != float("inf") else None


def _metric_rows(results, metrics_requested):
    rows = []
    for result in results:
        rows.append({
            "seed": int(result.seed),
            "metrics": {k: _json_number(result.metrics.get(k)) for k in metrics_requested},
            "seconds": float(result.seconds),
            "backend": result.backend,
        })
    aggregate = {}
    for key in metrics_requested:
        vals = [row["metrics"].get(key) for row in rows if row["metrics"].get(key) is not None]
        aggregate[key] = float(sum(vals) / len(vals)) if vals else None
    return aggregate, rows


def create_app(*, backend: str = "cuda_exact", device: str | None = None,
               max_concurrency: int = 1, panel_concurrency: int = 1,
               block: int = 1024, si_block: int = 1024):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise ImportError("install the server extra: pip install -e '.[server]'") from exc

    import torch

    app = FastAPI(title="FastClusterEval", version="0.1.0")
    semaphore = asyncio.Semaphore(max_concurrency)
    panel_semaphore = asyncio.Semaphore(max(1, int(panel_concurrency)))
    registry = DatasetRegistry()
    state = {"started_at": time.time(), "requests": 0, "active": 0}

    @app.on_event("startup")
    async def startup():
        if device and str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.init()

    @app.get("/health")
    async def health():
        cuda = bool(torch.cuda.is_available())
        return {
            "status": "ok", "backend": backend, "device": device,
            "cuda_available": cuda, "gpu_count": torch.cuda.device_count() if cuda else 0,
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
            "max_concurrency": max_concurrency, "requests": state["requests"],
            "active": state["active"],
        }

    @app.post("/v1/datasets/register")
    async def register_dataset(payload: dict[str, Any]):
        try:
            record = await asyncio.to_thread(registry.register, payload)
            return registry.info(record.dataset_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail={"type": type(exc).__name__, "message": str(exc)}) from exc

    @app.post("/v2/datasets/{dataset_id}/views/register")
    async def register_view(dataset_id: str, payload: dict[str, Any]):
        try:
            return await asyncio.to_thread(
                registry.register_view, dataset_id, str(payload["view_id"]), payload["row_indices"]
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail={"type": type(exc).__name__, "message": str(exc)}) from exc

    @app.get("/v1/datasets/{dataset_id}")
    async def dataset_info(dataset_id: str):
        try:
            return registry.info(dataset_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail={"type": type(exc).__name__, "message": str(exc)}) from exc

    @app.delete("/v1/datasets/{dataset_id}")
    async def unregister_dataset(dataset_id: str):
        registry.unregister(dataset_id)
        return {"status": "ok", "dataset_id": dataset_id}

    async def run(payload: dict[str, Any]):
        state["requests"] += 1
        async with semaphore:
            state["active"] += 1
            try:
                req = request_from_dict(payload)
                result = await asyncio.to_thread(
                    evaluate_batch, [req], backend=payload.get("backend", backend),
                    device=payload.get("device", device), max_concurrency=1,
                    block=int(payload.get("block", block)), si_block=int(payload.get("si_block", si_block)),
                    cugraph_resolution=payload.get("cugraph_resolution"),
                )
                return result["datasets"][0]
            except Exception as exc:
                raise HTTPException(status_code=400, detail={"type": type(exc).__name__, "message": str(exc)}) from exc
            finally:
                state["active"] -= 1

    @app.post("/v1/evaluate")
    async def evaluate_one(payload: dict[str, Any]):
        return await run(payload)

    @app.post("/v1/evaluate_batch")
    async def evaluate_many(payload: dict[str, Any]):
        try:
            reqs = [request_from_dict(d) for d in payload["datasets"]]
            async with semaphore:
                state["requests"] += len(reqs)
                state["active"] += 1
                try:
                    return await asyncio.to_thread(
                        evaluate_batch, reqs, backend=payload.get("backend", backend),
                        device=payload.get("device", device),
                        max_concurrency=int(payload.get("dataset_concurrency", 1)),
                        block=int(payload.get("block", block)), si_block=int(payload.get("si_block", si_block)),
                        cugraph_resolution=payload.get("cugraph_resolution"),
                    )
                finally:
                    state["active"] -= 1
        except Exception as exc:
            raise HTTPException(status_code=400, detail={"type": type(exc).__name__, "message": str(exc)}) from exc

    @app.post("/v1/evaluate_teacher_step")
    async def evaluate_teacher_step(payload: dict[str, Any]):
        """Evaluate all panels on multiple resident D/A views in one request."""
        try:
            dataset_id = str(payload["dataset_id"])
            panels = payload["selected_matrix_indices"]
            views = payload.get("views") or {"full": {}}
            results = {}
            async with semaphore:
                state["requests"] += len(views)
                state["active"] += 1
                try:
                    for view_id, spec in views.items():
                        rows = spec.get("row_indices") if isinstance(spec, dict) else spec
                        matrix, truth = registry.view(dataset_id, rows)
                        req = DatasetInput(
                            dataset_id=f"{dataset_id}:{view_id}",
                            matrix=matrix,
                            truth_labels=truth,
                            selected_matrix_indices=panels,
                            metrics=tuple(payload.get("metrics", ("NMI", "ARI", "SI"))),
                            seeds=tuple(int(x) for x in payload.get("seeds", (0, 1, 2))),
                        )
                        batch = await asyncio.to_thread(
                            evaluate_batch, [req], backend=payload.get("backend", backend),
                            device=payload.get("device", device), max_concurrency=1,
                            block=int(payload.get("block", block)),
                            si_block=int(payload.get("si_block", si_block)),
                            cugraph_resolution=payload.get("cugraph_resolution"),
                        )
                        results[str(view_id)] = batch["datasets"][0]
                finally:
                    state["active"] -= 1
            return {"dataset_id": dataset_id, "backend": payload.get("backend", backend), "views": results}
        except Exception as exc:
            raise HTTPException(status_code=400, detail={"type": type(exc).__name__, "message": str(exc)}) from exc

    @app.post("/v2/teacher/step/screen")
    @app.post("/v2/teacher/step/verify")
    async def evaluate_teacher_step_v2(payload: dict[str, Any]):
        """Evaluate one incremental teacher step.

        A view is clustered once. Optional projections are scored using the
        resulting partition, matching the teacher's audit-half protocol.
        The controller decides which repetitions are new; this endpoint never
        expands a request into repetitions that were not supplied.
        """
        try:
            dataset_id = str(payload["dataset_id"])
            panels = payload["selected_matrix_indices"]
            views = payload.get("views") or {"full": {}}
            requested_metrics = tuple(payload.get("metrics", ("NMI", "ARI", "SI")))
            seeds = tuple(int(x) for x in payload.get("seeds", (0, 1, 2)))
            selected_backend = payload.get("backend", backend)
            result_views = {}
            started = time.perf_counter()
            async with semaphore:
                state["requests"] += 1
                state["active"] += 1
                try:
                    for view_id, spec in views.items():
                        spec = spec if isinstance(spec, dict) else {"row_indices": spec}
                        if "view_id" in spec:
                            rows = registry.view_rows(dataset_id, str(spec["view_id"]))
                        else:
                            rows = spec.get("row_indices")
                        matrix, truth = registry.view(dataset_id, rows)
                        projections = spec.get("projections", {})
                        view_seeds = tuple(int(x) for x in spec.get("seeds", seeds))
                        view_started = time.perf_counter()
                        panel_results = []
                        panel_specs = list(DatasetInput(
                            dataset_id=f"{dataset_id}:{view_id}",
                            matrix=matrix,
                            truth_labels=truth,
                            selected_matrix_indices=panels,
                            metrics=requested_metrics,
                            seeds=view_seeds,
                        ).panels().items())

                        async def evaluate_panel(panel_id, indices):
                            async with panel_semaphore:
                                evaluated = await asyncio.to_thread(
                                    evaluate,
                                    matrix[:, indices], truth, seeds=view_seeds,
                                    backend=selected_backend,
                                    device=payload.get("device", device),
                                    block=int(payload.get("block", block)),
                                    si_block=int(payload.get("si_block", si_block)),
                                    cugraph_resolution=payload.get("cugraph_resolution"),
                                )
                            return panel_id, indices, evaluated

                        evaluated_panels = await asyncio.gather(
                            *(evaluate_panel(panel_id, indices) for panel_id, indices in panel_specs)
                        )
                        for panel_id, indices, evaluated in evaluated_panels:
                            aggregate, seed_rows = _metric_rows(evaluated, requested_metrics)
                            projection_rows = {}
                            for projection_id, projection_spec in projections.items():
                                projection_spec = projection_spec if isinstance(projection_spec, dict) else {"row_indices": projection_spec}
                                projection_indices = projection_spec.get("row_indices")
                                if isinstance(projection_indices, str):
                                    projection_indices = registry.view_rows(dataset_id, projection_indices)
                                projection_indices = np.asarray(projection_indices, dtype=np.int64).reshape(-1)
                                if projection_indices.size and (projection_indices.min() < 0 or projection_indices.max() >= len(truth)):
                                    raise ValueError(f"{dataset_id}/{view_id}/{projection_id}: projection outside view bounds")
                                projection_seed_rows = []
                                for result in evaluated:
                                    projected = metrics(
                                        truth[projection_indices], result.labels[projection_indices],
                                        result.evaluated[projection_indices],
                                        device=payload.get("device", device) if selected_backend != "reference" else None,
                                        backend="cuda" if selected_backend != "reference" else "cpu",
                                        si_block=int(payload.get("si_block", si_block)),
                                    )
                                    projection_seed_rows.append({
                                        "seed": int(result.seed),
                                        "metrics": {k: _json_number(projected.get(k)) for k in requested_metrics},
                                        "backend": result.backend,
                                    })
                                projection_aggregate = {}
                                for key in requested_metrics:
                                    vals = [r["metrics"].get(key) for r in projection_seed_rows if r["metrics"].get(key) is not None]
                                    projection_aggregate[key] = float(sum(vals) / len(vals)) if vals else None
                                projection_rows[str(projection_id)] = {
                                    "metrics": projection_aggregate,
                                    "seeds": projection_seed_rows,
                                }
                            panel_results.append({
                                "panel_id": str(panel_id),
                                "selected_matrix_indices": indices,
                                "metrics": aggregate,
                                "seeds": seed_rows,
                                "projections": projection_rows,
                            })
                        result_views[str(view_id)] = {
                            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                            "panels": panel_results,
                            "seconds": float(time.perf_counter() - view_started),
                        }
                finally:
                    state["active"] -= 1
            return {
                "schema_version": "2.0",
                "dataset_id": dataset_id,
                "backend": selected_backend,
                "views": result_views,
                "seconds": float(time.perf_counter() - started),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail={"type": type(exc).__name__, "message": str(exc)}) from exc

    return app


def main():
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="Run a resident FastClusterEval GPU service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--backend", choices=["reference", "cuda_exact", "cuda_graph", "cuda_cugraph", "cuda_proxy"], default="cuda_exact")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-concurrency", type=int, default=1)
    ap.add_argument("--panel-concurrency", type=int, default=1)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--si-block", type=int, default=1024)
    a = ap.parse_args()
    app = create_app(backend=a.backend, device=a.device, max_concurrency=a.max_concurrency,
                     panel_concurrency=a.panel_concurrency,
                     block=a.block, si_block=a.si_block)
    uvicorn.run(app, host=a.host, port=a.port, workers=1)


if __name__ == "__main__":
    main()
