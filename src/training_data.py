from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from .config import DEFAULT_MARKERS, DEFAULT_SPLIT_RATIOS, RANDOM_STATE
    from .data_loading import TrialRecord, load_trial_sequence
    from .metadata import SEX_LABELS, SEX_NAMES, participant_sex_map
    from .preprocessing import (
        FeatureStandardizer,
        dataframe_to_sequence,
        preprocess_motion_dataframe,
    )
    from .splitting import SPLIT_NAMES, split_participants
    from .windowing import create_labeled_windows, create_window_metadata
except ImportError:  # pragma: no cover - supports direct imports from notebooks
    from config import DEFAULT_MARKERS, DEFAULT_SPLIT_RATIOS, RANDOM_STATE
    from data_loading import TrialRecord, load_trial_sequence
    from metadata import SEX_LABELS, SEX_NAMES, participant_sex_map
    from preprocessing import FeatureStandardizer, dataframe_to_sequence, preprocess_motion_dataframe
    from splitting import SPLIT_NAMES, split_participants
    from windowing import create_labeled_windows, create_window_metadata


TASK_ACTIVITY = "activity"
TASK_GAIT_TYPE = "gait_type"
TASK_SEX = "sex"
TASK_PARTICIPANT_ID = "participant_id"
TASK_ENVIRONMENT_GAIT_TYPE = "environment_gait_type"

SPLIT_WITHIN_PARTICIPANT = "within_participant"
SPLIT_PARTICIPANT_HOLDOUT = "participant_holdout"


@dataclass(frozen=True)
class LabelEncoding:
    task: str
    class_names: tuple[str, ...]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def name_to_label(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.class_names)}


def make_trial_table(
    records: Sequence[TrialRecord],
    task: str,
    metadata: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, LabelEncoding]:
    """Create a trial-level table with encoded target labels."""
    rows = []
    sex_by_participant = participant_sex_map(metadata) if metadata is not None else {}

    for record in records:
        target_name = target_for_record(record, task=task, sex_by_participant=sex_by_participant)
        if target_name is None:
            continue

        rows.append(
            {
                "participant_id": record.participant_id,
                "session": record.session,
                "environment": record.environment,
                "activity": record.activity,
                "condition": record.condition,
                "protocol": record.protocol,
                "trial": record.trial,
                "path": record.path,
                "target_name": target_name,
            }
        )

    if not rows:
        raise ValueError(f"No labelled records were created for task {task!r}.")

    table = pd.DataFrame(rows)
    class_names = _class_names_for_task(task, table)
    name_to_label = {name: index for index, name in enumerate(class_names)}
    table["target_label"] = table["target_name"].map(name_to_label).astype(int)

    return table.reset_index(drop=True), LabelEncoding(task=task, class_names=class_names)


def target_for_record(
    record: TrialRecord,
    task: str,
    sex_by_participant: dict[str, str] | None = None,
) -> str | None:
    if task == TASK_ACTIVITY:
        return record.activity
    if task == TASK_GAIT_TYPE:
        return f"{record.activity}_{record.condition}"
    if task == TASK_ENVIRONMENT_GAIT_TYPE:
        return f"{record.environment}_{record.activity}_{record.condition}"
    if task == TASK_SEX:
        if sex_by_participant is None:
            raise ValueError("Sex metadata is required for the sex classification task.")
        sex = sex_by_participant.get(record.participant_id)
        return None if sex is None else SEX_NAMES[SEX_LABELS[sex]]
    if task == TASK_PARTICIPANT_ID:
        return record.participant_id
    raise ValueError(f"Unknown task: {task}")


def split_trial_table(
    table: pd.DataFrame,
    strategy: str = SPLIT_WITHIN_PARTICIPANT,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Add a ``split`` column to a trial table.

    ``within_participant`` keeps every participant represented in train/val/test
    by splitting their trials. ``participant_holdout`` assigns whole
    participants to one split, which is useful for harder gait/sex validation
    but invalid for participant identification.
    """
    if strategy == SPLIT_WITHIN_PARTICIPANT:
        return _split_within_participant(table, ratios=ratios, random_state=random_state)
    if strategy == SPLIT_PARTICIPANT_HOLDOUT:
        return _split_participant_holdout(table, ratios=ratios, random_state=random_state)
    raise ValueError(f"Unknown split strategy: {strategy}")


def summarise_splits(table: pd.DataFrame) -> pd.DataFrame:
    """Return participant/trial/window-free counts per split and target."""
    return (
        table.groupby(["split", "target_name"], observed=True)
        .agg(
            trials=("path", "count"),
            participants=("participant_id", "nunique"),
        )
        .reset_index()
        .sort_values(["split", "target_name"])
    )


def fit_standardizer_from_table(
    table: pd.DataFrame,
    markers: Sequence[str] = DEFAULT_MARKERS,
) -> FeatureStandardizer:
    """Fit a feature standardizer on already-centred training frames."""
    return FeatureStandardizer.fit(
        _iter_preprocessed_frames(table, markers=markers, standardizer=None, drop_time=False)
    )


def build_window_arrays(
    table: pd.DataFrame,
    standardizer: FeatureStandardizer,
    window_size: int,
    step_size: int,
    markers: Sequence[str] = DEFAULT_MARKERS,
    max_windows: int | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build Keras-ready ``X, y`` arrays and per-window metadata."""
    windows_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    metadata_rows: list[dict] = []

    for _, row in table.reset_index(drop=True).iterrows():
        raw = load_trial_sequence(_row_to_record_proxy(row), markers=markers, include_time=True)
        processed = preprocess_motion_dataframe(raw, standardizer=standardizer)
        sequence = dataframe_to_sequence(processed)
        windows, labels = create_labeled_windows(
            sequence,
            label=int(row["target_label"]),
            window_size=window_size,
            step_size=step_size,
        )

        if windows.shape[0] == 0:
            continue

        windows_list.append(windows)
        labels_list.append(labels)
        metadata_rows.extend(
            create_window_metadata(
                n_frames=sequence.shape[0],
                window_size=window_size,
                step_size=step_size,
                participant_id=row["participant_id"],
                session=row["session"],
                environment=row["environment"],
                activity=row["activity"],
                condition=row["condition"],
                target_name=row["target_name"],
                target_label=int(row["target_label"]),
                source_path=Path(row["path"]).as_posix(),
            )
        )

    if not windows_list:
        raise ValueError("No windows were created. Try a smaller window_size.")

    x = np.concatenate(windows_list, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(labels_list, axis=0).astype(np.int64, copy=False)
    metadata = pd.DataFrame(metadata_rows)

    if max_windows is not None and x.shape[0] > max_windows:
        x, y, metadata = sample_windows(
            x,
            y,
            metadata,
            max_windows=max_windows,
            random_state=random_state,
        )

    return x, y, metadata.reset_index(drop=True)


def build_split_arrays(
    table: pd.DataFrame,
    standardizer: FeatureStandardizer,
    window_size: int,
    step_size: int,
    markers: Sequence[str] = DEFAULT_MARKERS,
    max_windows_per_split: int | None = None,
    random_state: int = RANDOM_STATE,
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    arrays = {}
    for split_name in SPLIT_NAMES:
        split_table = table.loc[table["split"] == split_name]
        arrays[split_name] = build_window_arrays(
            split_table,
            standardizer=standardizer,
            window_size=window_size,
            step_size=step_size,
            markers=markers,
            max_windows=max_windows_per_split,
            random_state=random_state,
        )
    return arrays


def class_weight_dict(y: np.ndarray) -> dict[int, float]:
    """Compute inverse-frequency class weights for sparse labels."""
    labels, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    n_classes = len(labels)
    return {
        int(label): float(total / (n_classes * count))
        for label, count in zip(labels, counts, strict=True)
    }


def sample_windows(
    x: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    max_windows: int,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Downsample windows with a roughly class-balanced draw."""
    if max_windows <= 0:
        raise ValueError("max_windows must be positive.")
    if x.shape[0] <= max_windows:
        return x, y, metadata

    rng = np.random.default_rng(random_state)
    labels = np.unique(y)
    per_class = max(1, max_windows // len(labels))
    chosen: list[np.ndarray] = []

    for label in labels:
        indices = np.flatnonzero(y == label)
        n_take = min(per_class, len(indices))
        chosen.append(rng.choice(indices, size=n_take, replace=False))

    selected = np.concatenate(chosen)
    remainder = max_windows - selected.shape[0]
    if remainder > 0:
        available = np.setdiff1d(np.arange(y.shape[0]), selected, assume_unique=False)
        if available.size:
            extra = rng.choice(available, size=min(remainder, available.size), replace=False)
            selected = np.concatenate([selected, extra])

    rng.shuffle(selected)
    return x[selected], y[selected], metadata.iloc[selected].reset_index(drop=True)


def _split_within_participant(
    table: pd.DataFrame,
    ratios: Sequence[float],
    random_state: int,
) -> pd.DataFrame:
    result = table.copy()
    result["split"] = ""
    rng = np.random.default_rng(random_state)

    for _, participant_rows in result.groupby("participant_id", sort=True):
        for _, target_rows in participant_rows.groupby("target_name", sort=True):
            indices = target_rows.index.to_numpy(copy=True)
            rng.shuffle(indices)
            assignments = _assign_indices(indices, ratios)
            for split_name, split_indices in assignments.items():
                result.loc[split_indices, "split"] = split_name

    empty = result["split"] == ""
    if empty.any():
        raise RuntimeError("Some rows were not assigned to a split.")

    return result


def _split_participant_holdout(
    table: pd.DataFrame,
    ratios: Sequence[float],
    random_state: int,
) -> pd.DataFrame:
    if table["target_name"].nunique() == table["participant_id"].nunique():
        raise ValueError(
            "participant_holdout is invalid for participant_id classification: "
            "validation/test participants would be unseen classes."
        )

    participant_targets = table.groupby("participant_id")["target_name"].nunique()
    if (participant_targets == 1).all():
        participant_splits = _stratified_participant_holdout(table, ratios, random_state)
    else:
        participant_splits = split_participants(
            table["participant_id"].tolist(),
            ratios=ratios,
            random_state=random_state,
        )

    participant_to_split = {
        participant: split_name
        for split_name, participants in participant_splits.items()
        for participant in participants
    }

    result = table.copy()
    result["split"] = result["participant_id"].map(participant_to_split)
    return result


def _stratified_participant_holdout(
    table: pd.DataFrame,
    ratios: Sequence[float],
    random_state: int,
) -> dict[str, list[str]]:
    participant_labels = (
        table.groupby("participant_id")["target_name"]
        .first()
        .reset_index()
        .sort_values(["target_name", "participant_id"])
    )

    split_sets = {split_name: [] for split_name in SPLIT_NAMES}
    for _, group in participant_labels.groupby("target_name", sort=True):
        group_split = split_participants(
            group["participant_id"].tolist(),
            ratios=ratios,
            random_state=random_state,
        )
        for split_name in SPLIT_NAMES:
            split_sets[split_name].extend(group_split[split_name])

    return {split_name: sorted(values) for split_name, values in split_sets.items()}


def _assign_indices(indices: np.ndarray, ratios: Sequence[float]) -> dict[str, np.ndarray]:
    counts = _allocate_counts(len(indices), ratios)
    result = {}
    start = 0
    for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
        stop = start + count
        result[split_name] = indices[start:stop]
        start = stop
    return result


def _allocate_counts(n_items: int, ratios: Sequence[float]) -> list[int]:
    ratios_array = np.asarray(ratios, dtype=float)
    ratios_array = ratios_array / ratios_array.sum()
    raw = ratios_array * n_items
    counts = np.floor(raw).astype(int)

    remainder = n_items - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind="stable")
        for index in order[:remainder]:
            counts[index] += 1

    return counts.tolist()


def _iter_preprocessed_frames(
    table: pd.DataFrame,
    markers: Sequence[str],
    standardizer: FeatureStandardizer | None,
    drop_time: bool,
) -> Iterable[pd.DataFrame]:
    for _, row in table.reset_index(drop=True).iterrows():
        raw = load_trial_sequence(_row_to_record_proxy(row), markers=markers, include_time=True)
        yield preprocess_motion_dataframe(
            raw,
            standardizer=standardizer,
            drop_time=drop_time,
        )


def _row_to_record_proxy(row: pd.Series) -> TrialRecord:
    return TrialRecord(
        participant_id=str(row["participant_id"]),
        session=str(row["session"]),
        environment=str(row["environment"]),
        activity=str(row["activity"]),
        label=int(row.get("target_label", 0)),
        condition=str(row["condition"]),
        protocol=str(row["protocol"]),
        trial=str(row["trial"]),
        processing_stage="Post_Process",
        path=Path(row["path"]),
    )


def _class_names_for_task(task: str, table: pd.DataFrame) -> tuple[str, ...]:
    if task == TASK_SEX:
        return tuple(SEX_NAMES[index] for index in sorted(SEX_NAMES))
    return tuple(sorted(table["target_name"].dropna().unique().tolist()))
