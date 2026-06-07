from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

try:
    from .config import ACTIVITY_LABELS, DATA_DIR, DEFAULT_MARKERS
except ImportError:  # pragma: no cover - supports direct imports from notebooks
    from config import ACTIVITY_LABELS, DATA_DIR, DEFAULT_MARKERS


AXES = ("X", "Y", "Z")
METADATA_KEYS = {"FrameNumber", "FirstFrame", "PointFrequency", "AnalogFrequency"}


@dataclass(frozen=True)
class TrialRecord:
    participant_id: str
    session: str
    environment: str
    activity: str
    label: int
    condition: str
    protocol: str
    trial: str
    processing_stage: str
    path: Path

    @property
    def record_id(self) -> str:
        return self.path.with_suffix("").as_posix()

    def to_dict(self, base_dir: Path | None = None) -> dict:
        item = asdict(self)
        item["path"] = _path_for_manifest(self.path, base_dir)
        return item


@dataclass(frozen=True)
class CsvHeader:
    columns: tuple[str, ...]
    units: dict[str, str]
    metadata: dict[str, str]
    events: dict[str, tuple[float, ...]]
    header_row: int
    data_start_row: int
    frame_count: int


def discover_trials(
    data_dir: Path = DATA_DIR,
    stage: str = "Post_Process",
    activities: Iterable[str] | None = ("walk", "run"),
    environments: Iterable[str] | None = None,
) -> list[TrialRecord]:
    """Find motion CSV trials and parse participant/session labels from paths."""
    data_dir = Path(data_dir)
    activity_filter = _normalised_filter(activities)
    environment_filter = _normalised_filter(environments)

    records: list[TrialRecord] = []
    pattern = f"*/Session*/*/*/{stage}/*.csv"

    for path in data_dir.glob(pattern):
        record = parse_trial_path(path, data_dir=data_dir)
        if record is None:
            continue
        if activity_filter is not None and record.activity not in activity_filter:
            continue
        if environment_filter is not None and record.environment not in environment_filter:
            continue
        records.append(record)

    return sorted(
        records,
        key=lambda item: (
            item.participant_id,
            item.session,
            item.environment,
            item.activity,
            item.trial,
            item.path.name,
        ),
    )


def parse_trial_path(path: Path, data_dir: Path = DATA_DIR) -> TrialRecord | None:
    """Parse a dataset CSV path into a structured trial record."""
    path = Path(path)

    try:
        parts = path.relative_to(data_dir).parts
    except ValueError:
        parts = path.parts

    if len(parts) < 6:
        return None

    participant_id, session, protocol, trial, processing_stage = parts[:5]
    activity = _infer_activity(protocol, trial, path.stem)
    environment = _infer_environment(protocol)

    if activity is None or environment is None:
        return None

    condition = _infer_condition(trial, path.stem)

    return TrialRecord(
        participant_id=participant_id,
        session=session,
        environment=environment,
        activity=activity,
        label=ACTIVITY_LABELS[activity],
        condition=condition,
        protocol=protocol,
        trial=trial,
        processing_stage=processing_stage,
        path=path,
    )


def records_to_dataframe(records: Sequence[TrialRecord], base_dir: Path | None = None) -> pd.DataFrame:
    """Represent discovered trials as a tabular manifest."""
    return pd.DataFrame([record.to_dict(base_dir=base_dir) for record in records])


def parse_cgm_csv_header(path: Path) -> CsvHeader:
    """
    Read the custom CSV header exported with the CGM dataset.

    The post-processed CSV has a point section followed by analog rows with a
    different width. The FrameNumber metadata is therefore used later to read
    only marker frames.
    """
    metadata: dict[str, str] = {}
    events: dict[str, tuple[float, ...]] = {}

    with Path(path).open(newline="") as file:
        reader = csv.reader(file)
        for row_idx, row in enumerate(reader):
            if not row:
                continue

            key = row[0].strip()
            if key == "Time":
                labels = row
                units_row = next(reader)
                axes_row = next(reader)
                columns = _flatten_columns(labels, axes_row)
                units = {
                    column: unit.strip()
                    for column, unit in zip(columns, units_row, strict=False)
                }
                frame_count = _frame_count_from_metadata(metadata)
                return CsvHeader(
                    columns=tuple(columns),
                    units=units,
                    metadata=metadata,
                    events=events,
                    header_row=row_idx,
                    data_start_row=row_idx + 3,
                    frame_count=frame_count,
                )

            if key in METADATA_KEYS and len(row) > 1:
                metadata[key] = row[1].strip()
            elif key:
                values = tuple(float(value) for value in row[1:] if value.strip())
                events[key] = values

    raise ValueError(f"Could not find a Time header row in {path}.")


def read_motion_csv(
    path: Path,
    markers: Sequence[str] | None = DEFAULT_MARKERS,
    include_time: bool = True,
    required_markers: bool = True,
) -> pd.DataFrame:
    """
    Read point-coordinate frames from a post-processed motion CSV.

    Parameters
    ----------
    path:
        CSV file from a ``Post_Process`` directory.
    markers:
        Marker names to keep. ``None`` reads all flattened columns.
    include_time:
        Keep the ``Time`` column in the returned dataframe.
    required_markers:
        Raise an error when any requested marker axis is missing.
    """
    header = parse_cgm_csv_header(path)
    selected_columns = _selected_columns(header, markers, include_time, required_markers)
    column_to_index = {column: idx for idx, column in enumerate(header.columns)}
    selected_indices = sorted(column_to_index[column] for column in selected_columns)

    df = pd.read_csv(
        path,
        skiprows=header.data_start_row,
        header=None,
        nrows=header.frame_count,
        usecols=selected_indices,
        dtype=float,
    )
    df.columns = [header.columns[index] for index in selected_indices]

    return df.loc[:, selected_columns]


def load_trial_sequence(
    trial: TrialRecord,
    markers: Sequence[str] | None = DEFAULT_MARKERS,
    include_time: bool = False,
) -> pd.DataFrame:
    return read_motion_csv(trial.path, markers=markers, include_time=include_time)


def marker_columns(
    columns: Sequence[str],
    markers: Sequence[str],
    axes: Sequence[str] = AXES,
    required: bool = True,
) -> list[str]:
    """Return flattened ``MARKER_AXIS`` columns for selected markers."""
    available = set(columns)
    selected = [f"{marker}_{axis}" for marker in markers for axis in axes]
    missing = [column for column in selected if column not in available]

    if missing and required:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise KeyError(f"Missing marker columns: {preview}{suffix}")

    return [column for column in selected if column in available]


def _selected_columns(
    header: CsvHeader,
    markers: Sequence[str] | None,
    include_time: bool,
    required_markers: bool,
) -> list[str]:
    if markers is None:
        columns = list(header.columns)
        if not include_time:
            columns = [column for column in columns if column != "Time"]
        return columns

    columns = marker_columns(header.columns, markers, required=required_markers)
    if include_time:
        return ["Time", *columns]
    return columns


def _flatten_columns(labels: Sequence[str], axes: Sequence[str]) -> list[str]:
    columns: list[str] = []
    current_label = ""
    seen: dict[str, int] = {}

    for index, (label, axis) in enumerate(zip(labels, axes, strict=False)):
        label = label.strip()
        axis = axis.strip()

        if index == 0:
            name = "Time"
        else:
            if label:
                current_label = label
            name = f"{current_label}_{axis}" if axis else current_label

        count = seen.get(name, 0)
        seen[name] = count + 1
        columns.append(name if count == 0 else f"{name}_{count}")

    return columns


def _frame_count_from_metadata(metadata: dict[str, str]) -> int:
    try:
        return int(float(metadata["FrameNumber"]))
    except KeyError as exc:
        raise ValueError("CSV header does not contain FrameNumber metadata.") from exc


def _infer_activity(*parts: str) -> str | None:
    text = "_".join(parts).lower()
    if "walk" in text:
        return "walk"
    if "run" in text:
        return "run"
    return None


def _infer_environment(protocol: str) -> str | None:
    text = protocol.lower()
    if "treadmill" in text:
        return "treadmill"
    if "overground" in text:
        return "overground"
    return None


def _infer_condition(trial: str, stem: str) -> str:
    text = trial or stem
    parts = text.split("_")
    if len(parts) < 2:
        return "unknown"
    return parts[-1].lower()


def _normalised_filter(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    return {value.lower() for value in values}


def _path_for_manifest(path: Path, base_dir: Path | None) -> str:
    if base_dir is None:
        return path.as_posix()
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()
