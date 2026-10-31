# FastClusterEval

FastClusterEval is an independent evaluator for single-cell gene-panel
experiments. It has two goals:

1. preserve a CPU reference path compatible with the current Scanpy/Leiden
   evaluation protocol;
2. move the expensive panel preprocessing, kNN construction and metric kernels
   to CUDA so teacher search can evaluate many candidate panels in batches.

The project deliberately separates **reference results** from **accelerated
results**. A GPU Leiden implementation can produce a different partition from
`leidenalg`; those results are useful for screening only until the parity
benchmark passes.

## Install

```bash
# Windows/local development
<python-3.11> -m venv .venv
.venv/Scripts/python -m pip install -e ".[reference,dev]"

# Linux/GPU hosts
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[reference,dev]"
```

PyTorch must be installed with a CUDA build on GPU hosts. The optional RAPIDS
dependencies are not required for the first CUDA graph backend.

The local 4090 validation environment uses the repository-managed `.venv`.
It reuses the existing CUDA PyTorch/Scanpy wheels through a local
site-packages link, avoiding a second multi-gigabyte PyTorch installation.

## API

```python
import numpy as np
from fast_cluster_eval import evaluate

X = np.random.default_rng(0).poisson(1.0, size=(2000, 64)).astype(np.float32)
y = np.random.default_rng(1).integers(0, 8, size=len(X))

reference = evaluate(X, y, seeds=range(3), backend="reference")
cuda_graph = evaluate(X, y, seeds=range(3), backend="cuda_graph", device="cuda")

print(reference[0].metrics)
print(cuda_graph[0].metrics)
```

## Standard batch input/output

The stable interface is a list of independent dataset requests. Each request
contains the matrix, truth labels and one or more selected-panel index lists:

```json
{
  "schema_version": "1.0",
  "datasets": [
    {
      "dataset_id": "sample_001",
      "matrix": "sample_001.npy",
      "truth_labels": "sample_001_labels.npy",
      "selected_matrix_indices": {
        "teacher_64": [0, 4, 9, 12],
        "hvg_64": [1, 3, 8, 11]
      },
      "metrics": ["NMI", "ARI", "SI"],
      "seeds": [0, 1, 2]
    }
  ]
}
```

Run several datasets in one command:

```bash
fast-cluster-eval manifest.json \
  --output result.json \
  --backend cuda_graph \
  --device cuda \
  --max-concurrency 2
```

The output is always JSON-compatible and contains one atomic result object per
`dataset_id`. A failed dataset is recorded with `status: "error"` while other
datasets remain usable. Each panel contains aggregate metrics and per-seed
metrics, preserving the selected matrix indices used for the calculation.

For CUDA, set `max-concurrency` from available VRAM. Dataset-level concurrency
is intentionally explicit; unbounded panel-level concurrency causes memory
fragmentation and makes timing less reproducible.

## Resident GPU service

The evaluator can run as one long-lived GPU process. This keeps the CUDA
context and allocator warm across requests:

```bash
pip install -e ".[reference,server]"
fast-cluster-eval-server --host 127.0.0.1 --port 8765 \
  --backend cuda_exact --device cuda --max-concurrency 2
```

The service exposes `GET /health`, `POST /v1/evaluate`, and
`POST /v1/evaluate_batch`. The request body of `/v1/evaluate` is one dataset
object from the batch schema. For large matrices, use a server-local `.npy`,
`.npz`, or `.h5ad` path in `matrix` instead of sending nested JSON arrays.
Keep the service bound to localhost and use an SSH tunnel for remote access
unless the network is explicitly trusted.

`reference` runs normalization, kNN and Leiden through Scanpy. `cuda_exact`
keeps those exact reference labels and runs the NMI/ARI/SI kernels on CUDA;
this is the consistency mode. `cuda_graph` does normalization and blockwise
kNN on CUDA, then uses the same Leiden package on the resulting graph. Because
GPU kNN can make different tie/neighbor choices, `cuda_graph` is a fast
screening mode until its partition parity has been demonstrated on a target
dataset.

The metric kernels include NMI and ARI from a contingency matrix and a
blockwise Euclidean silhouette implementation. The blockwise implementation
does not allocate a full cell-by-cell distance matrix.

## Correctness protocol

Before using the CUDA backend for teacher decisions:

```bash
pytest
```

Then run the same panels through both backends and record:

- NMI, ARI and SI absolute differences;
- partition agreement and number of clusters;
- candidate ranking correlation and top-k recall;
- wall time, peak memory and GPU utilization.

The current project keeps NMI, ARI and SI as separate endpoints. A future fast
proxy may screen candidates, but it must be calibrated against the reference
evaluator on held-out source families and must not replace the frozen endpoint
used for RiGPS comparisons.

## Performance design

The main DAG is:

```text
panel batches → GPU normalize/log1p → blockwise GPU kNN → graph clustering →
GPU NMI/ARI/SI → top-k CPU exact reevaluation
```

The implementation avoids materializing an `n_cells × n_cells` matrix for kNN
or silhouette. `block` and `si_block` control the memory/performance tradeoff.
The expected end-to-end speedup depends on how much time Leiden consumes; GPU
metric kernels alone are not sufficient to accelerate the full pipeline.

The current custom CUDA DAG is:

```text
CUDA panel normalization/log1p
  → CUDA blockwise distance/top-k kNN
  → CUDA smooth-kNN sigma solve and edge weights
  → transfer only n_cells × k edge data
  → Leiden reference clustering
  → CUDA NMI/ARI/SI kernels
```

This keeps the large cell-by-cell matrix on the GPU and transfers only the
sparse edge list required by the current CPU Leiden implementation. The local
RTX 4090 smoke benchmark on a 2,048-cell, 32-gene panel measured roughly 19 s
for the Scanpy reference and roughly 1 s for the CUDA graph path. The CUDA
graph partition is not yet declared reference-equivalent; use `cuda_exact` for
final scores and `cuda_graph` for candidate screening until a dataset-level
parity report passes.
