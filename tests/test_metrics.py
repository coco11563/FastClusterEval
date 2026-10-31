import numpy as np
import torch

from fast_cluster_eval.metrics import adjusted_rand, normalized_mutual_information, sklearn_metrics


def test_exact_metrics_match_sklearn():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 5, 400)
    p = rng.integers(0, 7, 400)
    expected = sklearn_metrics(y, p)
    assert abs(normalized_mutual_information(y, p) - expected["NMI"]) < 1e-12
    assert abs(adjusted_rand(y, p) - expected["ARI"]) < 1e-12


def test_cuda_metrics_if_available():
    if not torch.cuda.is_available():
        return
    rng = np.random.default_rng(5)
    y = rng.integers(0, 4, 300)
    p = rng.integers(0, 6, 300)
    expected = sklearn_metrics(y, p)
    assert abs(normalized_mutual_information(y, p, "cuda") - expected["NMI"]) < 1e-10
    assert abs(adjusted_rand(y, p, "cuda") - expected["ARI"]) < 1e-10
