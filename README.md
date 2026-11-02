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

### Isolated RAPIDS environment on gpu17

The validated remote RAPIDS environment is kept separate from the training
environment at `/home/xiaomeng/FastClusterEval/rapids-venv`. It uses Python
3.11, `cudf-cu12==25.10.0`, `cugraph-cu12==25.10.1`, and
`cupy-cuda12x==14.2.0`. Use its `bin/rapids-python` wrapper; it preloads the
RAPIDS CUDA libraries before PyTorch to avoid a cuBLAS ABI collision.

```bash
/home/xiaomeng/FastClusterEval/rapids-venv/bin/rapids-python -c \
  'import torch, cudf, cugraph; print(torch.cuda.is_available(), cudf.__version__, cugraph.__version__)'
```

The wrapper currently reports eight A100 devices. This environment is for
cuGraph validation and GPU screening; reference-aligned final scores still use
`cuda_exact`.

## Quick start for another agent

```bash
git clone git@github.com:coco11563/FastClusterEval.git
cd FastClusterEval
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[reference,dev]"
```

The matrix convention is **cells × genes**. `truth_labels` has one entry per
cell. `selected_matrix_indices` is zero-based and refers to gene columns. A
single panel can be a list; multiple panels should be a mapping from panel ID
to index list.

Run the local tests first:

```bash
python -m pytest -q
```

For a CUDA host, verify the device before benchmarking:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python benchmarks/bench_4090.py
```

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
      "row_indices": "sample_001_rows_D.npy",
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

The referenced `.npy`, `.npz`, `.csv`, `.tsv`, or `.h5ad` paths are resolved
relative to the manifest file. A minimal manifest can therefore be generated
without embedding large matrices in JSON:

```json
{
  "schema_version": "1.0",
  "datasets": [
    {
      "dataset_id": "study_A",
      "matrix": "data/study_A.npy",
      "truth_labels": "data/study_A_labels.npy",
      "selected_matrix_indices": {"panel_64": [0, 2, 5, 9]},
      "metrics": ["NMI", "ARI", "SI"],
      "seeds": [0, 1, 2]
    }
  ]
}
```

The output is always JSON-compatible and contains one atomic result object per
`dataset_id`. A failed dataset is recorded with `status: "error"` while other
datasets remain usable. Each panel contains aggregate metrics and per-seed
metrics, preserving the selected matrix indices used for the calculation.

`row_indices` is optional. When present, the service evaluates only the
specified cells while keeping matrix column indices unchanged. The teacher
uses this for selection and audit views (D/A) without creating a separate
matrix for every panel.

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

For a remote service on `gpu_pub`:

```bash
ssh -N -L 8765:127.0.0.1:8765 gpu_pub
curl http://127.0.0.1:8765/health
```

For a remote service on `gpu17` using its A100 GPU:

```bash
ssh -N -L 8766:127.0.0.1:8766 gpu17
curl http://127.0.0.1:8766/health
```

The resident service returns one dataset result for `/v1/evaluate` and a
standard batch object for `/v1/evaluate_batch`. Undefined silhouette values are
serialized as JSON `null`, never as non-standard `NaN`.

### BioJev teacher-step API v2

For a greedy teacher, register the dataset once and send only the new
repetitions at each adaptive stage:

```http
POST /v1/datasets/register
POST /v2/teacher/step/screen
```

The v2 request contains `views`, `selected_matrix_indices`, and optional
per-view `seeds`. An audit view can contain `projections`; these are scored
using the partition produced on the full audit view, so audit halves do not
recluster independently:

```json
{
  "dataset_id": "teacher:W008",
  "views": {
    "sel_0": {"row_indices": [0, 2, 4], "seeds": [0]},
    "aud_0": {
      "row_indices": [10, 11, 12], "seeds": [100],
      "projections": {"h1": {"row_indices": [0, 2]}}
    }
  },
  "selected_matrix_indices": {"P": [1, 4], "0": [1, 4, 9]},
  "metrics": ["NMI", "ARI", "SI"],
  "backend": "cuda_graph"
}
```

`/v2/teacher/step/verify` uses the same schema for exact top-k or tie
verification. The controller must send only new selection repetitions and send
audit views once. The v2 endpoint reports per-view and per-panel timings and
keeps the v1 endpoints for compatibility. Exact services may set
`--panel-concurrency 2` or `4` after a local parity check; the GPU metric work
is still guarded by the service's CUDA execution order.

## Current remote GPU pool

The services bind to localhost on each server. Use an SSH tunnel from the
client machine, or call `127.0.0.1` directly when the agent runs on that
server.

| Server | Port | GPU role | Backend | Concurrency |
|---|---:|---|---|---:|
| `gpu_pub` | 8765 | A800 GPU0, limited free memory | `cuda_exact` | 2 |
| `gpu_pub` | 8773–8776 | A800 GPU0–3, shared slots | `cuda_graph` | 1 each |
| `gpu17` | 8766 | A100 GPU0 | `cuda_exact` | 2 |
| `gpu17` | 8767–8772 | A100 GPU1,2,3,4,5,7 | `cuda_graph` | 4 each |

Use `cuda_exact` for BioJev teacher decisions and final scores. It keeps the
CPU reference Scanpy/Leiden partition and moves NMI/ARI/SI kernels to CUDA, so
its results are reference-aligned. Use `cuda_graph` only for high-throughput
candidate-screening experiments: its Leiden partition can differ from the CPU
reference because GPU kNN ties and graph construction can differ. Send
finalists to `cuda_exact` before creating teacher targets. `cuda_cugraph` is an optional RAPIDS backend that
moves Leiden to GPU as well; it must pass a partition/metric parity report
before being used for teacher targets.

cuGraph's Leiden resolution is exposed separately as `cugraph_resolution` in
the Python, CLI, batch, and HTTP APIs. The Scanpy/leidenalg resolution is not
guaranteed to map numerically to cuGraph on a fuzzy kNN graph. Treat this
backend as a GPU screening path until a representative calibration report
has been recorded; use `cuda_exact` for reference-aligned final scores.

Example tunnels from a local client:

```bash
ssh -N -L 18765:127.0.0.1:8765 gpu_pub
ssh -N -L 18767:127.0.0.1:8767 gpu17
ssh -N -L 18768:127.0.0.1:8768 gpu17
ssh -N -L 18766:127.0.0.1:8766 gpu17
```

Check a service before sending work:

```bash
curl http://127.0.0.1:18767/health
```

The matrix path in an API request is resolved on the **server**, so large
matrices should first be placed on the target server as `.npy`, `.npz`, or
`.h5ad`. Nested JSON matrices are intended only for small smoke tests.

Example exact request:

```bash
curl -X POST http://127.0.0.1:18766/v1/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_id": "study_A",
    "matrix": "/data/eval/study_A.npy",
    "truth_labels": "/data/eval/study_A_labels.npy",
    "selected_matrix_indices": {
      "teacher_64": [0, 4, 9, 12],
      "hvg_64": [1, 3, 8, 11]
    },
    "metrics": ["NMI", "ARI", "SI"],
    "seeds": [0, 1, 2]
  }'
```

For a Python client, distribute screening requests round-robin across the
`gpu17` graph ports and send only selected finalists to the exact port:

```python
import itertools
import requests

GRAPH = itertools.cycle([
    "http://127.0.0.1:18767",
    "http://127.0.0.1:18768",
])
EXACT = "http://127.0.0.1:18766"

def screen(request_body):
    return requests.post(next(GRAPH) + "/v1/evaluate", json={
        **request_body, "backend": "cuda_graph"
    }, timeout=1800).json()

def verify(request_body):
    return requests.post(EXACT + "/v1/evaluate", json={
        **request_body, "backend": "cuda_exact"
    }, timeout=1800).json()
```

Do not send RiGPS benchmark data to these remote services. Use them for DISCO
teacher/validation inputs or other explicitly permitted datasets.

`reference` runs normalization, kNN and Leiden through Scanpy. `cuda_exact`
keeps those exact reference labels and runs the NMI/ARI/SI kernels on CUDA;
this is the consistency mode. `cuda_graph` does normalization and blockwise
kNN on CUDA, then uses the same Leiden package on the resulting graph. Because
GPU kNN can make different tie/neighbor choices, `cuda_graph` is a fast
screening mode until its partition parity has been demonstrated on a target
dataset. `cuda_cugraph` uses optional RAPIDS/cuGraph GPU Leiden and is a
throughput mode with the same parity requirement.

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
