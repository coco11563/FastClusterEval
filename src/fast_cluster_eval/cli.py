from __future__ import annotations

import argparse
import json
from pathlib import Path

from .batch import evaluate_batch
from .io import request_from_dict


def main():
    ap = argparse.ArgumentParser(description="FastClusterEval standardized batch evaluator")
    ap.add_argument("manifest", help="JSON manifest with a datasets list")
    ap.add_argument("--output", required=True, help="JSON output path")
    ap.add_argument("--backend", default="reference", choices=["reference", "cuda_exact", "cuda_graph", "cuda_cugraph", "cuda_proxy"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-concurrency", type=int, default=1)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--si-block", type=int, default=1024)
    a = ap.parse_args()
    manifest_path = Path(a.manifest)
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    reqs = [request_from_dict(d, manifest_path.parent) for d in doc["datasets"]]
    out = evaluate_batch(reqs, backend=a.backend, device=a.device,
                         max_concurrency=a.max_concurrency, block=a.block, si_block=a.si_block)
    Path(a.output).write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
