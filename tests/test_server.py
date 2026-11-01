import numpy as np
import pytest


def test_api_serializes_undefined_si_as_null():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fast_cluster_eval.server import create_app

    app = create_app(backend="cuda_exact", device="cpu")
    client = TestClient(app)
    payload = {
        "dataset_id": "constant",
        "matrix": np.ones((8, 4), dtype=float).tolist(),
        "truth_labels": (np.arange(8) % 2).tolist(),
        "selected_matrix_indices": {"p": [0, 1, 2]},
        "metrics": ["NMI", "ARI", "SI"],
        "seeds": [0],
    }
    response = client.post("/v1/evaluate", json=payload)
    assert response.status_code == 200
    assert response.json()["panels"][0]["metrics"]["SI"] is None


def test_resident_teacher_step_row_view():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fast_cluster_eval.server import create_app

    app = create_app(backend="reference", device="cpu")
    client = TestClient(app)
    x = np.ones((12, 4), dtype=float)
    y = np.arange(12) % 2
    register = client.post("/v1/datasets/register", json={
        "dataset_id": "resident",
        "matrix": x.tolist(),
        "truth_labels": y.tolist(),
    })
    assert register.status_code == 200
    response = client.post("/v1/evaluate_teacher_step", json={
        "dataset_id": "resident",
        "views": {"D": {"row_indices": list(range(0, 12, 2))}},
        "selected_matrix_indices": {"p": [0, 1, 2]},
        "metrics": ["NMI", "ARI", "SI"],
        "seeds": [0],
        "backend": "reference",
    })
    assert response.status_code == 200
    assert response.json()["views"]["D"]["shape"] == [6, 4]


def test_v2_teacher_step_reuses_full_audit_partition_for_projections():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fast_cluster_eval.evaluator import evaluate
    from fast_cluster_eval.metrics import metrics
    from fast_cluster_eval.server import create_app

    rng = np.random.default_rng(7)
    x = rng.poisson(2.0, size=(24, 6)).astype(float)
    x[:, 0] += np.arange(24) % 3
    y = np.repeat(np.arange(3), 8)
    client = TestClient(create_app(backend="reference", device="cpu"))
    register = client.post("/v1/datasets/register", json={
        "dataset_id": "v2", "matrix": x.tolist(), "truth_labels": y.tolist(),
    })
    assert register.status_code == 200
    payload = {
        "dataset_id": "v2",
        "views": {
            "sel_0": {"row_indices": list(range(12)), "seeds": [0]},
            "aud_0": {
                "row_indices": list(range(12, 24)),
                "seeds": [100],
                "projections": {"h1": {"row_indices": [0, 2, 4, 6]}, "h2": {"row_indices": [1, 3, 5, 7]}},
            },
        },
        "selected_matrix_indices": {"P": [0, 1, 2], "0": [0, 1, 2, 3]},
        "metrics": ["NMI", "ARI", "SI"],
        "backend": "reference",
    }
    response = client.post("/v2/teacher/step/screen", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema_version"] == "2.0"
    assert set(body["views"]) == {"sel_0", "aud_0"}

    audit = x[12:24]
    audit_y = y[12:24]
    direct = evaluate(audit[:, [0, 1, 2, 3]], audit_y, seeds=[100], backend="reference")[0]
    expected = metrics(audit_y[[0, 2, 4, 6]], direct.labels[[0, 2, 4, 6]],
                       direct.evaluated[[0, 2, 4, 6]], backend="cpu")
    actual = body["views"]["aud_0"]["panels"][1]["projections"]["h1"]["metrics"]
    assert actual["NMI"] == pytest.approx(expected["NMI"])


def test_v2_cuda_exact_matches_reference():
    pytest.importorskip("fastapi")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from fastapi.testclient import TestClient
    from fast_cluster_eval.server import create_app

    rng = np.random.default_rng(13)
    x = rng.poisson(2.0, size=(20, 5)).astype(float)
    y = np.repeat(np.arange(2), 10)
    payload = {
        "dataset_id": "parity",
        "views": {"aud": {
            "row_indices": list(range(20)), "seeds": [3],
            "projections": {"h1": {"row_indices": list(range(0, 20, 2))},
                             "h2": {"row_indices": list(range(1, 20, 2))}},
        }},
        "selected_matrix_indices": {"P": [0, 1, 2], "0": [0, 1, 2, 3]},
        "metrics": ["NMI", "ARI", "SI"],
    }
    outputs = {}
    for backend, device in (("reference", "cpu"), ("cuda_exact", "cuda")):
        client = TestClient(create_app(backend=backend, device=device))
        register = client.post("/v1/datasets/register", json={
            "dataset_id": "parity", "matrix": x.tolist(), "truth_labels": y.tolist(),
        })
        assert register.status_code == 200
        request = {**payload, "backend": backend}
        response = client.post("/v2/teacher/step/verify", json=request)
        assert response.status_code == 200, response.text
        outputs[backend] = response.json()["views"]["aud"]["panels"]
    for ref, exact in zip(outputs["reference"], outputs["cuda_exact"]):
        for metric in ("NMI", "ARI", "SI"):
            if ref["metrics"][metric] is not None:
                assert exact["metrics"][metric] == pytest.approx(ref["metrics"][metric], abs=1e-6)
        for projection in ("h1", "h2"):
            assert exact["projections"][projection]["metrics"]["NMI"] == pytest.approx(
                ref["projections"][projection]["metrics"]["NMI"], abs=1e-6
            )
