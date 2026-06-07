from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import DEFAULT_MARKERS, DEFAULT_STEP_SIZE, DEFAULT_WINDOW_SIZE, RANDOM_STATE
from data_loading import discover_trials
from evaluation import save_window_and_recording_evaluation
from metadata import load_subject_metadata
from models import build_lstm_classifier
from preprocessing import FeatureStandardizer
from training_data import (
    SPLIT_WITHIN_PARTICIPANT,
    TASK_GAIT_TYPE,
    TASK_PARTICIPANT_ID,
    TASK_SEX,
    build_split_arrays,
    class_weight_dict,
    fit_standardizer_from_table,
    make_trial_table,
    split_trial_table,
    summarise_splits,
)


def main() -> None:
    args = parse_args()
    set_seeds(args.random_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_trials(environments=args.environments)
    metadata = load_subject_metadata()

    all_results = []
    for task in args.tasks:
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)

        table, encoding = make_trial_table(records, task=task, metadata=metadata)
        table = split_trial_table(
            table,
            strategy=args.split_strategy,
            random_state=args.random_state,
        )
        save_table(table, task_dir / "trial_split.csv")
        summarise_splits(table).to_csv(task_dir / "split_summary.csv", index=False)

        standardizer = fit_standardizer_from_table(
            table.loc[table["split"] == "train"],
            markers=DEFAULT_MARKERS,
        )
        (task_dir / "standardizer.json").write_text(
            json.dumps(standardizer.to_dict(), indent=2),
            encoding="utf-8",
        )

        for window_size in args.window_sizes:
            arrays = build_split_arrays(
                table,
                standardizer=standardizer,
                window_size=window_size,
                step_size=args.step_size,
                markers=DEFAULT_MARKERS,
                max_windows_per_split=args.max_windows_per_split,
                random_state=args.random_state,
            )

            x_train, y_train, train_meta = arrays["train"]
            x_val, y_val, val_meta = arrays["val"]
            x_test, y_test, test_meta = arrays["test"]

            window_dir = task_dir / f"window_{window_size}_step_{args.step_size}"
            window_dir.mkdir(parents=True, exist_ok=True)
            train_meta.to_csv(window_dir / "train_windows.csv", index=False)
            val_meta.to_csv(window_dir / "val_windows.csv", index=False)
            test_meta.to_csv(window_dir / "test_windows.csv", index=False)

            for units, dropout, learning_rate in product(
                args.lstm_units,
                args.dropouts,
                args.learning_rates,
            ):
                run_name = run_id(window_size, args.step_size, units, dropout, learning_rate)
                run_dir = window_dir / run_name
                run_dir.mkdir(parents=True, exist_ok=True)

                model = build_lstm_classifier(
                    window_size=window_size,
                    n_features=x_train.shape[-1],
                    n_classes=encoding.n_classes,
                    lstm_units=units,
                    dense_units=args.dense_units,
                    dropout=dropout,
                    learning_rate=learning_rate,
                    bidirectional=args.bidirectional,
                )

                callbacks = [
                    EarlyStopping(
                        monitor="val_loss",
                        patience=args.patience,
                        restore_best_weights=True,
                    ),
                    ModelCheckpoint(
                        filepath=run_dir / "best_model.keras",
                        monitor="val_loss",
                        save_best_only=True,
                    ),
                ]

                history = model.fit(
                    x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    class_weight=class_weight_dict(y_train),
                    callbacks=callbacks,
                    verbose=args.verbose,
                )

                pd.DataFrame(history.history).to_csv(run_dir / "history.csv", index=False)
                model.save(run_dir / "last_model.keras")

                y_proba = model.predict(x_test, batch_size=args.batch_size)
                evaluation = save_window_and_recording_evaluation(
                    metadata=test_meta,
                    y_true=y_test,
                    y_proba=y_proba,
                    label_names=encoding.class_names,
                    output_dir=run_dir,
                )

                payload = {
                    "task": task,
                    "class_names": list(encoding.class_names),
                    "split_strategy": args.split_strategy,
                    "window_size": window_size,
                    "step_size": args.step_size,
                    "lstm_units": list(units),
                    "dense_units": args.dense_units,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "bidirectional": args.bidirectional,
                    "epochs": len(history.history["loss"]),
                    "n_train_windows": int(x_train.shape[0]),
                    "n_val_windows": int(x_val.shape[0]),
                    "n_test_windows": int(x_test.shape[0]),
                    "evaluation": evaluation,
                }

                (run_dir / "metrics.json").write_text(
                    json.dumps(to_jsonable(payload), indent=2),
                    encoding="utf-8",
                )
                all_results.append(flatten_result(payload, run_dir))
                print(
                    f"[{task}] {run_name}: "
                    f"window={evaluation['window']['metrics']} "
                    f"recording={evaluation['recording']['metrics']}"
                )

    pd.DataFrame(all_results).to_csv(output_dir / "results_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[TASK_GAIT_TYPE, TASK_SEX, TASK_PARTICIPANT_ID],
        choices=[TASK_GAIT_TYPE, TASK_SEX, TASK_PARTICIPANT_ID],
    )
    parser.add_argument("--split-strategy", default=SPLIT_WITHIN_PARTICIPANT)
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[DEFAULT_WINDOW_SIZE])
    parser.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--lstm-units", nargs="+", type=parse_units, default=[(64,)])
    parser.add_argument("--dense-units", type=int, default=None)
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.3])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.001])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--max-windows-per-split", type=int, default=None)
    parser.add_argument("--environments", nargs="+", default=None)
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "models" / "lstm_experiments")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def parse_units(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def run_id(
    window_size: int,
    step_size: int,
    units: tuple[int, ...],
    dropout: float,
    learning_rate: float,
) -> str:
    units_text = "x".join(str(unit) for unit in units)
    return (
        f"lstm_{units_text}"
        f"_win_{window_size}"
        f"_step_{step_size}"
        f"_drop_{dropout:g}"
        f"_lr_{learning_rate:g}"
    )


def save_table(table: pd.DataFrame, path: Path) -> None:
    serialisable = table.copy()
    serialisable["path"] = serialisable["path"].map(lambda item: Path(item).as_posix())
    serialisable.to_csv(path, index=False)


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def flatten_result(payload: dict, run_dir: Path) -> dict:
    window_metrics = payload["evaluation"]["window"]["metrics"]
    recording_metrics = payload["evaluation"]["recording"]["metrics"]
    return {
        "task": payload["task"],
        "run_dir": run_dir.as_posix(),
        "window_size": payload["window_size"],
        "step_size": payload["step_size"],
        "lstm_units": "x".join(str(item) for item in payload["lstm_units"]),
        "dense_units": payload["dense_units"],
        "dropout": payload["dropout"],
        "learning_rate": payload["learning_rate"],
        "bidirectional": payload["bidirectional"],
        "epochs": payload["epochs"],
        "n_train_windows": payload["n_train_windows"],
        "n_val_windows": payload["n_val_windows"],
        "n_test_windows": payload["n_test_windows"],
        **{f"window_{key}": value for key, value in window_metrics.items()},
        **{f"recording_{key}": value for key, value in recording_metrics.items()},
    }


if __name__ == "__main__":
    main()
