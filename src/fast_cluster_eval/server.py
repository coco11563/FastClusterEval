"""Resident HTTP service for GPU cluster evaluation."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .batch import evaluate_batch
from .io import DatasetInput, request_from_dict
from .registry import DatasetRegistry


def create_app(*, backend: str = "cuda_exact", device: str | None = None,
               max_concurrency: int = 1, block: int = 1024, si_block: int = 1024):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise ImportError("install the server extra: pip install -e '.[server]'") from exc

    import torch

    app = FastAPI(title="FastClusterEval", version="0.1.0")
    semaphore = asyncio.Semaphore(max_concurrency)
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
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--si-block", type=int, default=1024)
    a = ap.parse_args()
    app = create_app(backend=a.backend, device=a.device, max_concurrency=a.max_concurrency,
                     block=a.block, si_block=a.si_block)
    uvicorn.run(app, host=a.host, port=a.port, workers=1)


if __name__ == "__main__":
    main()
