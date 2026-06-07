from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute core classification metrics for sparse integer labels."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
) -> dict:
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=list(label_names),
        output_dict=True,
        zero_division=0,
    )


def make_window_predictions(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_names: Sequence[str],
) -> pd.DataFrame:
    """Create a per-window prediction table from metadata and model probabilities."""
    y_pred = np.argmax(y_proba, axis=1)
    result = metadata.reset_index(drop=True).copy()
    result["y_true"] = y_true.astype(int)
    result["y_true_name"] = [label_names[index] for index in y_true.astype(int)]
    result["y_pred"] = y_pred.astype(int)
    result["y_pred_name"] = [label_names[index] for index in y_pred.astype(int)]
    result["confidence"] = y_proba.max(axis=1)

    for class_index, class_name in enumerate(label_names):
        result[f"proba_{class_name}"] = y_proba[:, class_index]

    return result


def aggregate_recording_predictions(
    window_predictions: pd.DataFrame,
    label_names: Sequence[str],
    group_column: str = "source_path",
) -> pd.DataFrame:
    """
    Aggregate window-level predictions into recording-level predictions.

    The predicted class is selected by averaging class probabilities across all
    windows belonging to a recording.
    """
    probability_columns = [f"proba_{class_name}" for class_name in label_names]
    required = {group_column, "y_true", *probability_columns}
    missing = required - set(window_predictions.columns)
    if missing:
        raise KeyError(f"Missing columns for recording aggregation: {sorted(missing)}")

    rows = []
    for recording_id, group in window_predictions.groupby(group_column, sort=True):
        probabilities = group.loc[:, probability_columns].mean(axis=0).to_numpy(dtype=float)
        y_pred = int(np.argmax(probabilities))
        y_true_values = group["y_true"].unique()
        if len(y_true_values) != 1:
            raise ValueError(
                f"Recording {recording_id} has inconsistent true labels: {y_true_values}"
            )
        y_true = int(y_true_values[0])

        row = {
            group_column: recording_id,
            "n_windows": int(len(group)),
            "participant_id": _first_or_none(group, "participant_id"),
            "session": _first_or_none(group, "session"),
            "environment": _first_or_none(group, "environment"),
            "activity": _first_or_none(group, "activity"),
            "condition": _first_or_none(group, "condition"),
            "y_true": y_true,
            "y_true_name": label_names[y_true],
            "y_pred": y_pred,
            "y_pred_name": label_names[y_pred],
            "confidence": float(probabilities[y_pred]),
        }
        for class_index, class_name in enumerate(label_names):
            row[f"proba_{class_name}"] = float(probabilities[class_index])
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_prediction_frame(
    predictions: pd.DataFrame,
    label_names: Sequence[str],
) -> dict:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    y_pred = predictions["y_pred"].to_numpy(dtype=int)
    return {
        "metrics": evaluate_predictions(y_true, y_pred),
        "classification_report": classification_report_dict(y_true, y_pred, label_names),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(label_names))),
        ).tolist(),
    }


def save_evaluation_artifacts(
    predictions: pd.DataFrame,
    label_names: Sequence[str],
    output_dir: Path,
    prefix: str,
) -> dict:
    """Save predictions, metrics JSON, confusion matrix CSV and PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    evaluation = evaluate_prediction_frame(predictions, label_names)

    (output_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(_jsonable(evaluation), indent=2),
        encoding="utf-8",
    )

    matrix = np.asarray(evaluation["confusion_matrix"], dtype=int)
    pd.DataFrame(matrix, index=label_names, columns=label_names).to_csv(
        output_dir / f"{prefix}_confusion_matrix.csv"
    )
    save_confusion_matrix_plot(
        matrix,
        label_names=label_names,
        output_path=output_dir / f"{prefix}_confusion_matrix.png",
        title=f"{prefix.replace('_', ' ').title()} Confusion Matrix",
    )

    return evaluation


def save_window_and_recording_evaluation(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_names: Sequence[str],
    output_dir: Path,
) -> dict:
    """Save evaluation artifacts for windows and aggregated recordings."""
    window_predictions = make_window_predictions(
        metadata=metadata,
        y_true=y_true,
        y_proba=y_proba,
        label_names=label_names,
    )
    recording_predictions = aggregate_recording_predictions(
        window_predictions,
        label_names=label_names,
    )

    window_eval = save_evaluation_artifacts(
        window_predictions,
        label_names=label_names,
        output_dir=output_dir,
        prefix="window",
    )
    recording_eval = save_evaluation_artifacts(
        recording_predictions,
        label_names=label_names,
        output_dir=output_dir,
        prefix="recording",
    )

    comparison = compare_window_recording_metrics(window_eval, recording_eval)
    comparison.to_csv(Path(output_dir) / "window_vs_recording_metrics.csv", index=False)

    return {
        "window": window_eval,
        "recording": recording_eval,
        "comparison": comparison.to_dict(orient="records"),
    }


def compare_window_recording_metrics(window_eval: dict, recording_eval: dict) -> pd.DataFrame:
    rows = []
    for level, evaluation in [
        ("window", window_eval),
        ("recording", recording_eval),
    ]:
        row = {"level": level}
        row.update(evaluation["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
) -> None:
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        display_labels=list(label_names),
        values_format="d",
        xticks_rotation=45,
    )
    plt.tight_layout()
    plt.show()


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    label_names: Sequence[str],
    output_path: Path,
    title: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=_figure_size(label_names))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=list(label_names),
    )
    display.plot(ax=ax, values_format="d", xticks_rotation=45, colorbar=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _figure_size(label_names: Sequence[str]) -> tuple[float, float]:
    size = max(6.0, min(16.0, 0.42 * len(label_names) + 4.0))
    return size, size


def _first_or_none(frame: pd.DataFrame, column: str):
    if column not in frame.columns:
        return None
    values = frame[column].dropna().unique()
    if len(values) == 0:
        return None
    return values[0]


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
