from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from .config import PELVIS_MARKERS
except ImportError:  # pragma: no cover - supports direct imports from notebooks
    from config import PELVIS_MARKERS


AXES = ("X", "Y", "Z")


@dataclass(frozen=True)
class FeatureStandardizer:
    """Column-wise z-score standardizer fitted on training frames only."""

    mean_: pd.Series
    scale_: pd.Series
    feature_columns: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        frames: pd.DataFrame | Iterable[pd.DataFrame],
        feature_columns: Sequence[str] | None = None,
    ) -> "FeatureStandardizer":
        frame_iterator = _as_frame_iterator(frames)
        first_frame = next(frame_iterator, None)
        if first_frame is None:
            raise ValueError("At least one dataframe is required to fit a standardizer.")

        columns = tuple(feature_columns or infer_feature_columns(first_frame))
        if not columns:
            raise ValueError("No feature columns were provided or inferred.")

        total = np.zeros(len(columns), dtype=float)
        total_sq = np.zeros(len(columns), dtype=float)
        count = np.zeros(len(columns), dtype=float)

        for frame in _prepend(first_frame, frame_iterator):
            _ensure_columns(frame, columns)
            values = frame.loc[:, columns].to_numpy(dtype=float)
            finite = np.isfinite(values)
            safe_values = np.where(finite, values, 0.0)
            total += safe_values.sum(axis=0)
            total_sq += (safe_values * safe_values).sum(axis=0)
            count += finite.sum(axis=0)

        if np.any(count == 0):
            missing = [columns[index] for index in np.where(count == 0)[0]]
            raise ValueError(f"Columns contain no finite values: {missing}")

        mean = total / count
        variance = np.maximum(total_sq / count - mean * mean, 0.0)
        scale = np.sqrt(variance)
        scale[scale == 0.0] = 1.0

        return cls(
            mean_=pd.Series(mean, index=columns),
            scale_=pd.Series(scale, index=columns),
            feature_columns=columns,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        _ensure_columns(frame, self.feature_columns)
        result = frame.copy()
        result.loc[:, self.feature_columns] = (
            result.loc[:, self.feature_columns] - self.mean_
        ) / self.scale_
        return result

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != len(self.feature_columns):
            raise ValueError(
                "The last array dimension must match the number of fitted feature columns."
            )
        return (values - self.mean_.to_numpy()) / self.scale_.to_numpy()

    def to_dict(self) -> dict:
        return {
            "feature_columns": list(self.feature_columns),
            "mean": self.mean_.to_dict(),
            "scale": self.scale_.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FeatureStandardizer":
        columns = tuple(payload["feature_columns"])
        return cls(
            mean_=pd.Series(payload["mean"], index=columns, dtype=float),
            scale_=pd.Series(payload["scale"], index=columns, dtype=float),
            feature_columns=columns,
        )


def infer_feature_columns(frame: pd.DataFrame, include_time: bool = False) -> list[str]:
    """Infer numeric model features from a motion dataframe."""
    columns = list(frame.select_dtypes(include=[np.number]).columns)
    if include_time:
        return columns
    return [column for column in columns if column != "Time"]


def coordinate_columns(
    frame_or_columns: pd.DataFrame | Sequence[str],
    markers: Sequence[str] | None = None,
    axes: Sequence[str] = AXES,
) -> list[str]:
    """Return flattened coordinate columns such as ``LASI_X`` and ``LASI_Y``."""
    columns = (
        list(frame_or_columns.columns)
        if isinstance(frame_or_columns, pd.DataFrame)
        else list(frame_or_columns)
    )
    available = set(columns)

    if markers is None:
        suffixes = tuple(f"_{axis}" for axis in axes)
        return [column for column in columns if column.endswith(suffixes)]

    selected = [f"{marker}_{axis}" for marker in markers for axis in axes]
    missing = [column for column in selected if column not in available]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise KeyError(f"Missing coordinate columns: {preview}{suffix}")
    return selected


def trim_zero_frames(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    atol: float = 0.0,
) -> pd.DataFrame:
    """Drop leading and trailing frames where all selected features are zero."""
    columns = list(feature_columns or infer_feature_columns(frame))
    _ensure_columns(frame, columns)

    values = frame.loc[:, columns].to_numpy(dtype=float)
    non_zero = np.isfinite(values) & (np.abs(values) > atol)
    mask = non_zero.any(axis=1)

    if not mask.any():
        return frame.iloc[0:0].copy()

    start = int(np.argmax(mask))
    stop = len(mask) - int(np.argmax(mask[::-1]))
    return frame.iloc[start:stop].reset_index(drop=True)


def compute_pelvis_center(
    frame: pd.DataFrame,
    pelvis_markers: Sequence[str] = PELVIS_MARKERS,
) -> pd.DataFrame:
    """Compute per-frame pelvis centre from LASI/RASI/LPSI/RPSI markers."""
    center = pd.DataFrame(index=frame.index)
    for axis in AXES:
        columns = [f"{marker}_{axis}" for marker in pelvis_markers]
        _ensure_columns(frame, columns)
        center[axis] = frame.loc[:, columns].mean(axis=1)
    return center


def center_around_pelvis(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    pelvis_markers: Sequence[str] = PELVIS_MARKERS,
) -> pd.DataFrame:
    """Translate selected XYZ coordinates so the pelvis centre is at the origin."""
    columns = list(feature_columns or coordinate_columns(frame))
    _ensure_columns(frame, columns)

    result = frame.copy()
    pelvis_center = compute_pelvis_center(result, pelvis_markers=pelvis_markers)

    for axis in AXES:
        axis_columns = [column for column in columns if column.endswith(f"_{axis}")]
        if axis_columns:
            result.loc[:, axis_columns] = result.loc[:, axis_columns].sub(
                pelvis_center[axis],
                axis=0,
            )

    return result


def preprocess_motion_dataframe(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    pelvis_markers: Sequence[str] = PELVIS_MARKERS,
    trim_zeros: bool = True,
    center: bool = True,
    standardizer: FeatureStandardizer | None = None,
    drop_time: bool = True,
) -> pd.DataFrame:
    """
    Apply the first motion preprocessing baseline.

    The intended order is: trim zero padding, center coordinates around the
    pelvis, then apply z-score standardization learned on the training split.
    """
    columns = list(feature_columns or coordinate_columns(frame))
    result = frame.copy()

    if trim_zeros:
        result = trim_zero_frames(result, columns)

    if center:
        result = center_around_pelvis(
            result,
            feature_columns=columns,
            pelvis_markers=pelvis_markers,
        )

    if standardizer is not None:
        result = standardizer.transform(result)

    if drop_time and "Time" in result.columns:
        result = result.drop(columns="Time")

    return result


def dataframe_to_sequence(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> np.ndarray:
    """Convert a preprocessed dataframe to ``(n_frames, n_features)``."""
    columns = list(feature_columns or infer_feature_columns(frame))
    _ensure_columns(frame, columns)
    return frame.loc[:, columns].to_numpy(dtype=np.float32)


def fit_standardizer(
    frames: pd.DataFrame | Iterable[pd.DataFrame],
    feature_columns: Sequence[str] | None = None,
) -> FeatureStandardizer:
    return FeatureStandardizer.fit(frames, feature_columns=feature_columns)


def _as_frame_iterator(frames: pd.DataFrame | Iterable[pd.DataFrame]):
    if isinstance(frames, pd.DataFrame):
        return iter([frames])
    return iter(frames)


def _prepend(first_frame: pd.DataFrame, frames):
    yield first_frame
    yield from frames


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise KeyError(f"Missing dataframe columns: {preview}{suffix}")
