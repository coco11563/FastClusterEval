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
