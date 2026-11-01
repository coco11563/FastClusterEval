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
