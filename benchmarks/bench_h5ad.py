import argparse
import time

import anndata as ad
import numpy as np

from fast_cluster_eval import evaluate
from fast_cluster_eval.metrics import adjusted_rand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("--genes", type=int, default=32)
    ap.add_argument("--backend", action="append", default=["reference", "cuda_exact", "cuda_graph", "cuda_cugraph"])
    a = ap.parse_args()
    A = ad.read_h5ad(a.h5ad)
    X = A.X[:, :a.genes]
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype="float32")
    y = A.obs["major"].astype("category").cat.codes.to_numpy()
    reference_labels = None
    for backend in dict.fromkeys(a.backend):
        t = time.perf_counter()
        try:
            r = evaluate(X, y, seeds=(0,), backend=backend, device="cuda", block=512, si_block=512)[0]
            extra = {}
            if backend == "reference":
                reference_labels = r.labels
            elif reference_labels is not None:
                extra["partition_ARI_vs_reference"] = adjusted_rand(reference_labels, r.labels)
            print(backend, round(time.perf_counter() - t, 3), r.metrics, len(set(r.labels)), extra)
        except Exception as exc:
            print(backend, "ERROR", type(exc).__name__, str(exc)[:300])


if __name__ == "__main__":
    main()
