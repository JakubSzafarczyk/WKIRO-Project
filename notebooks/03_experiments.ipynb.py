# %% [markdown]
# # Eksperymenty LSTM: gait type, plec, identyfikacja uczestnika
#
# Ten notebook uruchamia ten sam pipeline dla trzech zadan:
# - `gait_type`: klasy ruchu z danych, np. `walk_slow`, `walk_fast`, `run_comfortable`,
# - `sex`: klasyfikacja plci uczestnika,
# - `participant_id`: identyfikacja konkretnego uczestnika.
#
# Domyslny split to `within_participant`: kazdy uczestnik jest dzielony proporcjonalnie
# na train/val/test, dzieki czemu zbiory sa rownomierne na poziomie uczestnikow.
# Dla twardszej ewaluacji gait/sex mozna zmienic na `participant_holdout`.

# %%
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import DEFAULT_MARKERS, DEFAULT_STEP_SIZE, RANDOM_STATE
from data_loading import discover_trials
from evaluation import classification_report_dict, evaluate_predictions
from metadata import load_subject_metadata
from models import build_lstm_classifier
from training_data import (
    SPLIT_PARTICIPANT_HOLDOUT,
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

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# %%
OUTPUT_DIR = PROJECT_ROOT / "models" / "lstm_experiments_notebook"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS_TO_RUN = [
    TASK_GAIT_TYPE,
    TASK_SEX,
    TASK_PARTICIPANT_ID,
]

SPLIT_STRATEGY = SPLIT_WITHIN_PARTICIPANT

WINDOW_SIZES = [50, 100]
STEP_SIZE = DEFAULT_STEP_SIZE

PARAM_GRID = [
    {"lstm_units": (32,), "dropout": 0.3, "learning_rate": 1e-3, "dense_units": None},
    {"lstm_units": (64,), "dropout": 0.3, "learning_rate": 1e-3, "dense_units": None},
    {"lstm_units": (64, 32), "dropout": 0.3, "learning_rate": 1e-3, "dense_units": 32},
]

EPOCHS = 20
BATCH_SIZE = 128
PATIENCE = 5

# Ustaw None dla pelnego treningu. Na pierwsze uruchomienie lepiej zostawic limit.
MAX_WINDOWS_PER_SPLIT = 8000

# %%
records = discover_trials()
metadata = load_subject_metadata()

print(f"Liczba prob CSV: {len(records)}")
print(f"Liczba uczestnikow: {len({record.participant_id for record in records})}")
metadata[["participant_id", "session", "sex", "age", "bodymass", "height"]].head()

# %%
def run_name(window_size: int, params: dict) -> str:
    units = "x".join(str(unit) for unit in params["lstm_units"])
    return (
        f"lstm_{units}"
        f"_win_{window_size}"
        f"_step_{STEP_SIZE}"
        f"_drop_{params['dropout']:g}"
        f"_lr_{params['learning_rate']:g}"
    )


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_trial_table(table: pd.DataFrame, path: Path) -> None:
    out = table.copy()
    out["path"] = out["path"].map(lambda item: Path(item).as_posix())
    out.to_csv(path, index=False)


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value

# %% [markdown]
# ## Podglad etykiet i splitow

# %%
for task in TASKS_TO_RUN:
    table, encoding = make_trial_table(records, task=task, metadata=metadata)
    table = split_trial_table(table, strategy=SPLIT_STRATEGY)
    print("\nTASK:", task)
    print("Klasy:", encoding.class_names)
    display(summarise_splits(table))

# %% [markdown]
# ## Trening modeli
#
# Wyniki dla kazdej konfiguracji trafia do:
# `models/lstm_experiments_notebook/<task>/window_<...>/<run_name>/`.

# %%
all_results = []

for task in TASKS_TO_RUN:
    task_dir = OUTPUT_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)

    table, encoding = make_trial_table(records, task=task, metadata=metadata)
    table = split_trial_table(table, strategy=SPLIT_STRATEGY)
    save_trial_table(table, task_dir / "trial_split.csv")
    summarise_splits(table).to_csv(task_dir / "split_summary.csv", index=False)

    train_table = table.loc[table["split"] == "train"]
    standardizer = fit_standardizer_from_table(train_table, markers=DEFAULT_MARKERS)
    save_json(task_dir / "standardizer.json", standardizer.to_dict())

    for window_size in WINDOW_SIZES:
        arrays = build_split_arrays(
            table,
            standardizer=standardizer,
            window_size=window_size,
            step_size=STEP_SIZE,
            markers=DEFAULT_MARKERS,
            max_windows_per_split=MAX_WINDOWS_PER_SPLIT,
            random_state=RANDOM_STATE,
        )

        x_train, y_train, train_meta = arrays["train"]
        x_val, y_val, val_meta = arrays["val"]
        x_test, y_test, test_meta = arrays["test"]

        window_dir = task_dir / f"window_{window_size}_step_{STEP_SIZE}"
        window_dir.mkdir(parents=True, exist_ok=True)
        train_meta.to_csv(window_dir / "train_windows.csv", index=False)
        val_meta.to_csv(window_dir / "val_windows.csv", index=False)
        test_meta.to_csv(window_dir / "test_windows.csv", index=False)

        print(
            f"\n{task}, window={window_size}: "
            f"train={x_train.shape}, val={x_val.shape}, test={x_test.shape}"
        )

        for params in PARAM_GRID:
            current_run = run_name(window_size, params)
            run_dir = window_dir / current_run
            run_dir.mkdir(parents=True, exist_ok=True)

            model = build_lstm_classifier(
                window_size=window_size,
                n_features=x_train.shape[-1],
                n_classes=encoding.n_classes,
                lstm_units=params["lstm_units"],
                dense_units=params["dense_units"],
                dropout=params["dropout"],
                learning_rate=params["learning_rate"],
            )

            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=PATIENCE,
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
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                class_weight=class_weight_dict(y_train),
                callbacks=callbacks,
                verbose=1,
            )

            pd.DataFrame(history.history).to_csv(run_dir / "history.csv", index=False)
            model.save(run_dir / "last_model.keras")

            y_pred = np.argmax(model.predict(x_test, batch_size=BATCH_SIZE), axis=1)
            metrics = evaluate_predictions(y_test, y_pred)
            report = classification_report_dict(y_test, y_pred, encoding.class_names)

            payload = {
                "task": task,
                "class_names": list(encoding.class_names),
                "split_strategy": SPLIT_STRATEGY,
                "window_size": window_size,
                "step_size": STEP_SIZE,
                "params": params,
                "epochs_done": len(history.history["loss"]),
                "n_train_windows": int(x_train.shape[0]),
                "n_val_windows": int(x_val.shape[0]),
                "n_test_windows": int(x_test.shape[0]),
                "metrics": metrics,
                "classification_report": report,
            }
            save_json(run_dir / "metrics.json", jsonable(payload))

            result_row = {
                "task": task,
                "run_dir": run_dir.as_posix(),
                "window_size": window_size,
                "step_size": STEP_SIZE,
                "lstm_units": "x".join(str(v) for v in params["lstm_units"]),
                "dense_units": params["dense_units"],
                "dropout": params["dropout"],
                "learning_rate": params["learning_rate"],
                "epochs_done": len(history.history["loss"]),
                "n_train_windows": int(x_train.shape[0]),
                "n_val_windows": int(x_val.shape[0]),
                "n_test_windows": int(x_test.shape[0]),
                **metrics,
            }
            all_results.append(result_row)
            print(current_run, metrics)

results = pd.DataFrame(all_results).sort_values(["task", "f1_macro"], ascending=[True, False])
results.to_csv(OUTPUT_DIR / "results_summary.csv", index=False)
results

# %% [markdown]
# ## Najlepsze konfiguracje per zadanie

# %%
results.sort_values("f1_macro", ascending=False).groupby("task").head(3)
