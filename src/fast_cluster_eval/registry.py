"""Resident server-side dataset registry for repeated panel evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_array


@dataclass
class DatasetRecord:
    dataset_id: str
    matrix: Any
    truth_labels: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    matrix_path: str | None = None
    truth_path: str | None = None
    rows: dict[str, np.ndarray] = field(default_factory=dict)


class DatasetRegistry:
    def __init__(self):
        self._records: dict[str, DatasetRecord] = {}

    def register(self, payload: dict[str, Any]) -> DatasetRecord:
        dataset_id = str(payload["dataset_id"])
        matrix_value = payload["matrix"]
        truth_value = payload["truth_labels"]
        matrix = load_array(matrix_value)
        truth = load_array(truth_value)
        if not hasattr(matrix, "shape"):
            matrix = np.asarray(matrix)
        truth = np.asarray(truth)
        if matrix.shape[0] != len(truth):
            raise ValueError(f"{dataset_id}: truth_labels length does not match matrix rows")
        record = DatasetRecord(
            dataset_id=dataset_id,
            matrix=matrix,
            truth_labels=truth,
            metadata=dict(payload.get("metadata", {})),
            matrix_path=str(matrix_value) if isinstance(matrix_value, (str, Path)) else None,
            truth_path=str(truth_value) if isinstance(truth_value, (str, Path)) else None,
        )
        self._records[dataset_id] = record
        return record

    def get(self, dataset_id: str) -> DatasetRecord:
        try:
            return self._records[dataset_id]
        except KeyError as exc:
            raise KeyError(f"dataset is not registered: {dataset_id}") from exc

    def register_view(self, dataset_id: str, view_id: str, row_indices: Any) -> dict[str, Any]:
        record = self.get(dataset_id)
        rows = np.asarray(load_array(row_indices), dtype=np.int64).reshape(-1)
        if rows.size and (rows.min() < 0 or rows.max() >= record.matrix.shape[0]):
            raise ValueError(f"{dataset_id}/{view_id}: row_indices outside dataset bounds")
        if len(np.unique(rows)) != len(rows):
            raise ValueError(f"{dataset_id}/{view_id}: row_indices contain duplicates")
        record.rows[str(view_id)] = rows
        return {
            "dataset_id": dataset_id,
            "view_id": str(view_id),
            "n_rows": int(rows.size),
            "row_hash": hashlib.sha256(rows.tobytes()).hexdigest(),
        }

    def view_rows(self, dataset_id: str, view_id: str) -> np.ndarray:
        record = self.get(dataset_id)
        key = str(view_id)
        rows = record.rows.get(key)
        if rows is None:
            try:
                rows = np.asarray(load_array(key), dtype=np.int64).reshape(-1)
            except Exception as exc:
                raise KeyError(f"view is not registered: {dataset_id}/{view_id}") from exc
            if rows.size and (rows.min() < 0 or rows.max() >= record.matrix.shape[0]):
                raise ValueError(f"{dataset_id}/{view_id}: row_indices outside dataset bounds")
            if len(np.unique(rows)) != len(rows):
                raise ValueError(f"{dataset_id}/{view_id}: row_indices contain duplicates")
            record.rows[key] = rows
        return rows

    def view(self, dataset_id: str, row_indices: Any = None) -> tuple[Any, np.ndarray]:
        record = self.get(dataset_id)
        if row_indices is None:
            return record.matrix, record.truth_labels
        if isinstance(row_indices, (str, Path)):
            key = str(row_indices)
            rows = record.rows.get(key)
            if rows is None:
                rows = np.asarray(load_array(row_indices), dtype=np.int64).reshape(-1)
                record.rows[key] = rows
        else:
            rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        if rows.size and (rows.min() < 0 or rows.max() >= record.matrix.shape[0]):
            raise ValueError(f"{dataset_id}: row_indices outside dataset bounds")
        return record.matrix[rows], record.truth_labels[rows]

    def info(self, dataset_id: str) -> dict[str, Any]:
        record = self.get(dataset_id)
        return {
            "dataset_id": record.dataset_id,
            "shape": [int(record.matrix.shape[0]), int(record.matrix.shape[1])],
            "metadata": record.metadata,
            "matrix_path": record.matrix_path,
            "truth_path": record.truth_path,
            "cached_views": len(record.rows),
        }

    def unregister(self, dataset_id: str) -> None:
        self._records.pop(dataset_id, None)
