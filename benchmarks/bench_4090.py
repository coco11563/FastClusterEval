import json, time
import numpy as np
import torch
from fast_cluster_eval import DatasetInput, evaluate, evaluate_batch

def main():
    rng = np.random.default_rng(11)
    n, g = 2048, 128
    X = rng.poisson(1.0, (n, g)).astype(np.float32)
    y = np.repeat(np.arange(8), n // 8)
    panels = {"p32": list(range(32)), "p64": list(range(64))}
    rows = {"torch": {"cuda": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    t = time.perf_counter()
    ref = evaluate(X[:, panels["p32"]], y, seeds=(0,), backend="reference")
    rows["reference_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    fast = evaluate(X[:, panels["p32"]], y, seeds=(0,), backend="cuda_graph", device="cuda", block=512, si_block=512)
    torch.cuda.synchronize()
    rows["cuda_graph_seconds"] = time.perf_counter() - t
    rows["reference_metrics"] = ref[0].metrics
    rows["cuda_metrics"] = fast[0].metrics
    t = time.perf_counter()
    exact = evaluate(X[:, panels["p32"]], y, seeds=(0,), backend="cuda_exact", device="cuda")
    torch.cuda.synchronize()
    rows["cuda_exact_seconds"] = time.perf_counter() - t
    rows["cuda_exact_metrics"] = exact[0].metrics
    reqs = [
        DatasetInput("d0", X, y, panels, seeds=(0,)),
        DatasetInput("d1", X * 1.0, y, {"p32": panels["p32"]}, seeds=(0,)),
    ]
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    batch = evaluate_batch(reqs, backend="cuda_graph", device="cuda", max_concurrency=1, block=512, si_block=512)
    torch.cuda.synchronize()
    rows["batch_seconds"] = time.perf_counter() - t
    rows["batch_summary"] = batch["summary"]
    rows["cuda_peak_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    print(json.dumps(rows, indent=2, allow_nan=True))

if __name__ == "__main__":
    main()
