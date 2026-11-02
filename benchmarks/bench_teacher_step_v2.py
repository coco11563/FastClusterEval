"""Compare the legacy resident teacher-step call pattern with v2 batching."""

from __future__ import annotations

import json
import argparse
import time

import numpy as np
from fastapi.testclient import TestClient

from fast_cluster_eval.server import create_app


def _panel_payload():
    return {"P": [0, 1, 2, 3], "0": [0, 1, 2, 4], "1": [0, 1, 2, 5], "2": [0, 1, 2, 6], "3": [0, 1, 2, 7]}


def _legacy(client, dataset_id, panels, rows, audit_half, backend):
    schedule = (2, 4, 6)
    audit_reps = 2
    requests = 0
    observed = {}
    started = time.perf_counter()
    for hi in schedule:
        # This reproduces the old resident adapter: every stage repeats every
        # selection and audit request instead of sending only new repetitions.
        for rep in range(max(schedule)):
            response = client.post("/v1/evaluate_teacher_step", json={
                "dataset_id": dataset_id,
                "views": {"sel": {"row_indices": rows}},
                "selected_matrix_indices": panels,
                "metrics": ["NMI", "ARI", "SI"],
                "seeds": [rep], "backend": backend,
            })
            response.raise_for_status()
            requests += 1
            if rep < 2:
                observed[rep] = response.json()["views"]["sel"]["panels"]
        for rep in range(audit_reps):
            response = client.post("/v1/evaluate_teacher_step", json={
                "dataset_id": dataset_id,
                "views": {
                    "sel": {"row_indices": rows},
                    "aud": {"row_indices": rows},
                    "aud_h1": {"row_indices": audit_half[0]},
                    "aud_h2": {"row_indices": audit_half[1]},
                },
                "selected_matrix_indices": panels,
                "metrics": ["NMI", "ARI", "SI"],
                "seeds": [100 + rep], "backend": backend,
            })
            response.raise_for_status()
            requests += 1
    return time.perf_counter() - started, requests, observed


def _v2(client, dataset_id, panels, rows, audit_half, backend):
    schedule = (2, 4, 6)
    audit_reps = 2
    requests = 0
    observed = {}
    started = time.perf_counter()
    for stage, hi in enumerate(schedule):
        views = {
            f"sel_{rep}": {"row_indices": rows, "seeds": [rep]}
            for rep in range(schedule[stage - 1] if stage else 0, hi)
        }
        if stage == 0:
            for rep in range(audit_reps):
                views[f"aud_{rep}"] = {
                    "row_indices": rows,
                    "seeds": [100 + rep],
                    "projections": {
                        "h1": {"row_indices": audit_half[0]},
                        "h2": {"row_indices": audit_half[1]},
                    },
                }
        response = client.post("/v2/teacher/step/screen", json={
            "dataset_id": dataset_id,
            "views": views,
            "selected_matrix_indices": panels,
            "metrics": ["NMI", "ARI", "SI"],
            "backend": backend,
        })
        response.raise_for_status()
        requests += 1
        for rep in range(schedule[stage - 1] if stage else 0, hi):
            observed[rep] = response.json()["views"][f"sel_{rep}"]["panels"]
    return time.perf_counter() - started, requests, observed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("reference", "cuda_exact", "cuda_graph"), default="reference")
    parser.add_argument("--panel-concurrency", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(19)
    x = rng.poisson(2.0, size=(96, 12)).astype(float)
    y = np.repeat(np.arange(4), 24)
    rows = list(range(96))
    half = [list(range(0, 48, 2)), list(range(1, 48, 2))]
    panels = _panel_payload()

    def run(fn):
        device = "cuda" if args.backend != "reference" else "cpu"
        client = TestClient(create_app(backend=args.backend, device=device,
                                       panel_concurrency=args.panel_concurrency))
        register = client.post("/v1/datasets/register", json={
            "dataset_id": "bench", "matrix": x.tolist(), "truth_labels": y.tolist(),
        })
        register.raise_for_status()
        return fn(client, "bench", panels, rows, half, args.backend)

    old_seconds, old_requests, old_values = run(_legacy)
    v2_seconds, v2_requests, v2_values = run(_v2)
    diffs = []
    for rep in sorted(old_values):
        old = {p["panel_id"]: p["metrics"] for p in old_values[rep]}
        new = {p["panel_id"]: p["metrics"] for p in v2_values[rep]}
        for panel_id in old:
            for metric in ("NMI", "ARI", "SI"):
                a, b = old[panel_id][metric], new[panel_id][metric]
                if a is not None and b is not None:
                    diffs.append(abs(float(a) - float(b)))
    result = {
        "old_seconds": old_seconds,
        "v2_seconds": v2_seconds,
        "speedup": old_seconds / v2_seconds if v2_seconds else None,
        "old_http_requests": old_requests,
        "v2_http_requests": v2_requests,
        "request_reduction": old_requests / v2_requests,
        "selection_metric_max_abs_diff": max(diffs) if diffs else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
