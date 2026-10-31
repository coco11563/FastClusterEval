"""Stable input/output schema for single and multi-dataset evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "1.0"
VALID_METRICS = ("NMI", "ARI", "SI")


def _panels(value) -> dict[str, list[int]]:
    if isinstance(value, dict):
        return {str(k): [int(i) for i in v] for k, v in value.items()}
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 1:
        return {"panel": [int(i) for i in arr.tolist()]}
    return {f"panel_{i}": [int(x) for x in row] for i, row in enumerate(value)}


@dataclass
class DatasetInput:
    dataset_id: str
    matrix: Any
    truth_labels: Any
    selected_matrix_indices: Any
    row_indices: Any = None
    metrics: tuple[str, ...] = VALID_METRICS
    seeds: tuple[int, ...] = (0, 1, 2)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "DatasetInput":
        n = int(self.matrix.shape[0])
        if len(self.truth_labels) != n:
            raise ValueError(f"{self.dataset_id}: truth_labels length {len(self.truth_labels)} != matrix rows {n}")
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty")
        invalid = set(self.metrics) - set(VALID_METRICS)
        if invalid:
            raise ValueError(f"{self.dataset_id}: unsupported metrics {sorted(invalid)}")
        p = _panels(self.selected_matrix_indices)
        g = int(self.matrix.shape[1])
        if self.row_indices is not None:
            rows = np.asarray(self.row_indices, dtype=np.int64).reshape(-1)
            if rows.size and (rows.min() < 0 or rows.max() >= n):
                raise ValueError(f"{self.dataset_id}: row_indices outside [0, {n})")
            if len(np.unique(rows)) != len(rows):
                raise ValueError(f"{self.dataset_id}: row_indices contain duplicates")
        for name, indices in p.items():
            if not indices:
                raise ValueError(f"{self.dataset_id}/{name}: selected panel is empty")
            if min(indices) < 0 or max(indices) >= g:
                raise ValueError(f"{self.dataset_id}/{name}: selected matrix index outside [0, {g})")
            if len(set(indices)) != len(indices) and not self.metadata.get("allow_duplicate_indices", False):
                raise ValueError(f"{self.dataset_id}/{name}: selected matrix indices contain duplicates")
        return self

    def panels(self) -> dict[str, list[int]]:
        self.validate()
        return _panels(self.selected_matrix_indices)


def load_array(value):
    """Load a standard manifest path (.npy/.npz/.csv/.h5ad) or pass arrays through."""
    if not isinstance(value, (str, Path)):
        return value
    path = Path(value)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        return z[z.files[0]]
    if path.suffix in (".csv", ".tsv"):
        return np.loadtxt(path, delimiter="\t" if path.suffix == ".tsv" else ",")
    if path.suffix == ".h5ad":
        import anndata as ad
        return ad.read_h5ad(path).X
    raise ValueError(f"unsupported matrix path: {path}")


def request_from_dict(d: dict[str, Any], base_dir: str | Path = ".") -> DatasetInput:
    base = Path(base_dir)
    matrix = load_array(d["matrix"] if not isinstance(d["matrix"], str) else base / d["matrix"])
    truth = load_array(d["truth_labels"] if not isinstance(d["truth_labels"], str) else base / d["truth_labels"])
    rows = load_array(d["row_indices"] if not isinstance(d.get("row_indices"), str) else base / d["row_indices"]) if d.get("row_indices") is not None else None
    if not hasattr(matrix, "shape"):
        matrix = np.asarray(matrix)
    if not hasattr(truth, "shape"):
        truth = np.asarray(truth)
    return DatasetInput(
        dataset_id=str(d["dataset_id"]),
        matrix=matrix,
        truth_labels=truth,
        selected_matrix_indices=d["selected_matrix_indices"],
        row_indices=rows,
        metrics=tuple(d.get("metrics", VALID_METRICS)),
        seeds=tuple(int(x) for x in d.get("seeds", (0, 1, 2))),
        metadata=dict(d.get("metadata", {})),
    ).validate()
