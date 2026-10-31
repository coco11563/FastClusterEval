import numpy as np

from fast_cluster_eval import DatasetInput, evaluate_batch


def test_standard_batch_schema_cpu():
    rng = np.random.default_rng(3)
    x = rng.poisson(1.0, (80, 12)).astype(np.float32)
    y = np.repeat(np.arange(4), 20)
    req = DatasetInput("d0", x, y, {"p0": [0, 1, 2], "p1": [3, 4, 5]}, seeds=(0,))
    out = evaluate_batch([req], backend="reference", max_concurrency=1)
    assert out["schema_version"] == "1.0"
    assert out["summary"] == {"count": 1, "ok": 1, "error": 0}
    assert len(out["datasets"][0]["panels"]) == 2


def test_dataset_failure_is_atomic():
    x = np.ones((10, 4), dtype=np.float32)
    good = DatasetInput("good", x, np.arange(10) % 2, {"p": [0, 1]}, seeds=(0,))
    bad = DatasetInput("bad", x, np.arange(9) % 2, {"p": [0, 1]}, seeds=(0,))
    out = evaluate_batch([good, bad], backend="reference")
    assert out["summary"] == {"count": 2, "ok": 1, "error": 1}


def test_row_view_evaluates_only_selected_cells():
    rng = np.random.default_rng(12)
    x = rng.poisson(1.0, (80, 12)).astype(np.float32)
    y = np.repeat(np.arange(4), 20)
    rows = np.arange(0, 80, 2)
    req = DatasetInput("view", x, y, {"p": [0, 1, 2]}, row_indices=rows, seeds=(0,))
    out = evaluate_batch([req], backend="reference", max_concurrency=1)
    assert out["summary"] == {"count": 1, "ok": 1, "error": 0}
    assert out["datasets"][0]["shape"] == [40, 12]


def test_cuda_exact_matches_reference_if_available():
    import torch
    if not torch.cuda.is_available():
        return
    rng = np.random.default_rng(9)
    x = rng.poisson(1.0, (256, 16)).astype(np.float32)
    y = np.arange(256) % 4
    from fast_cluster_eval import evaluate
    ref = evaluate(x, y, seeds=(0,), backend="reference")[0].metrics
    got = evaluate(x, y, seeds=(0,), backend="cuda_exact", device="cuda")[0].metrics
    assert abs(ref["NMI"] - got["NMI"]) < 1e-10
    assert abs(ref["ARI"] - got["ARI"]) < 1e-10
    assert abs(ref["SI"] - got["SI"]) < 1e-7
