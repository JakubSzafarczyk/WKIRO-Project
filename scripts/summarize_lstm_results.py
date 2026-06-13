from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANKING_METRIC = "recording_f1_macro"
PLOT_METRICS = [
    "window_accuracy",
    "recording_accuracy",
    "window_f1_macro",
    "recording_f1_macro",
]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = collect_results(output_dir)

    if not rows:
        raise SystemExit(f"No metrics.json files found under {output_dir}")

    summary = pd.DataFrame(rows)
    summary = add_display_columns(summary)
    summary = summary.sort_values(
        ["task", RANKING_METRIC, "recording_accuracy", "window_f1_macro"],
        ascending=[True, False, False, False],
    )

    summary_path = output_dir / "evaluation_summary.csv"
    summary.to_csv(summary_path, index=False)

    best = summary.groupby("task", group_keys=False).head(args.top_k)
    best_path = output_dir / f"best_runs_top{args.top_k}.csv"
    best.to_csv(best_path, index=False)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_summary_plots(summary, best, plots_dir)

    print(f"Saved {summary_path}")
    print(f"Saved {best_path}")
    print(f"Saved plots in {plots_dir}")
    print(best[["task", "experiment_name", "recording_accuracy", "recording_f1_macro"]])


def collect_results(output_dir: Path) -> list[dict]:
    rows = []
    for metrics_path in output_dir.rglob("metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        evaluation = payload.get("evaluation")
        if not evaluation:
            continue

        row = {
            "task": payload["task"],
            "experiment_name": payload.get("experiment_name", metrics_path.parent.name),
            "run_dir": metrics_path.parent.as_posix(),
            "window_size": payload["window_size"],
            "step_size": payload["step_size"],
            "lstm_units": "x".join(str(value) for value in payload["lstm_units"]),
            "n_lstm_layers": len(payload["lstm_units"]),
            "dense_units": payload["dense_units"],
            "dropout": payload["dropout"],
            "learning_rate": payload["learning_rate"],
            "bidirectional": payload["bidirectional"],
            "epochs": payload["epochs"],
            "n_train_windows": payload["n_train_windows"],
            "n_val_windows": payload["n_val_windows"],
            "n_test_windows": payload["n_test_windows"],
        }
        row.update(
            {
                f"window_{key}": value
                for key, value in evaluation["window"]["metrics"].items()
            }
        )
        row.update(
            {
                f"recording_{key}": value
                for key, value in evaluation["recording"]["metrics"].items()
            }
        )
        rows.append(row)
    return rows


def add_display_columns(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["dense_units_label"] = summary["dense_units"].fillna("none").astype(str)
    summary["model_label"] = summary.apply(model_label, axis=1)
    return summary


def model_label(row: pd.Series) -> str:
    return (
        f"{row['experiment_name']}\n"
        f"w={row['window_size']}, step={row['step_size']}, "
        f"LSTM={row['lstm_units']}, drop={row['dropout']}"
    )


def save_summary_plots(summary: pd.DataFrame, best: pd.DataFrame, plots_dir: Path) -> None:
    best_per_task = best.groupby("task", group_keys=False).head(1)
    plot_best_overall(best_per_task, plots_dir, "recording_accuracy")
    plot_best_overall(best_per_task, plots_dir, "recording_f1_macro")

    for task, task_df in summary.groupby("task", sort=True):
        safe_task = slugify(task)
        task_df = task_df.sort_values(RANKING_METRIC, ascending=False)
        plot_task_model_comparison(task_df, plots_dir / f"{safe_task}_model_comparison.png")
        plot_task_window_vs_recording(task_df, plots_dir / f"{safe_task}_window_vs_recording.png")
        plot_task_hyperparameters(task_df, plots_dir / f"{safe_task}_hyperparameter_effects.png")


def plot_best_overall(best: pd.DataFrame, plots_dir: Path, metric: str) -> None:
    data = best.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
    ax.bar(data["task"], data[metric], color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Najlepsze modele wedlug {metric.replace('_', ' ')}")
    annotate_bars(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / f"overall_best_{metric}.png", dpi=160)
    plt.close(fig)


def plot_task_model_comparison(task_df: pd.DataFrame, output_path: Path) -> None:
    data = task_df.sort_values(RANKING_METRIC, ascending=False)
    x = np.arange(len(data))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, len(data) * 1.5), 6))
    for index, metric in enumerate(PLOT_METRICS):
        offset = (index - 1.5) * width
        ax.bar(x + offset, data[metric], width=width, label=metric.replace("_", " "))

    ax.set_xticks(x)
    ax.set_xticklabels(data["model_label"], rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title(f"{data['task'].iloc[0]}: porownanie modeli")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_task_window_vs_recording(task_df: pd.DataFrame, output_path: Path) -> None:
    data = task_df.sort_values(RANKING_METRIC, ascending=False)
    x = np.arange(len(data))

    fig, axes = plt.subplots(1, 2, figsize=(max(11, len(data) * 1.3), 5), sharey=True)
    for ax, metric_name in zip(axes, ["accuracy", "f1_macro"], strict=True):
        ax.plot(x, data[f"window_{metric_name}"], marker="o", label="okna")
        ax.plot(x, data[f"recording_{metric_name}"], marker="s", label="nagrania")
        ax.set_xticks(x)
        ax.set_xticklabels(data["experiment_name"], rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(metric_name.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    fig.suptitle(f"{data['task'].iloc[0]}: okna vs cale nagrania")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_task_hyperparameters(task_df: pd.DataFrame, output_path: Path) -> None:
    data = task_df.sort_values(["window_size", "lstm_units", "dropout"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    grouped_bar(
        axes[0],
        data,
        group_column="window_size",
        metric=RANKING_METRIC,
        title="Rozmiar okna",
    )
    grouped_bar(
        axes[1],
        data,
        group_column="lstm_units",
        metric=RANKING_METRIC,
        title="Neurony / warstwy LSTM",
    )
    grouped_bar(
        axes[2],
        data,
        group_column="dropout",
        metric=RANKING_METRIC,
        title="Dropout",
    )

    fig.suptitle(f"{data['task'].iloc[0]}: wplyw hiperparametrow na recording F1 macro")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def grouped_bar(
    ax: plt.Axes,
    data: pd.DataFrame,
    group_column: str,
    metric: str,
    title: str,
) -> None:
    stats = (
        data.groupby(group_column, dropna=False)[metric]
        .agg(["mean", "max"])
        .reset_index()
        .sort_values(group_column)
    )
    labels = stats[group_column].astype(str)
    x = np.arange(len(stats))
    ax.bar(x - 0.18, stats["mean"], width=0.36, label="srednia")
    ax.bar(x + 0.18, stats["max"], width=0.36, label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()


def annotate_bars(ax: plt.Axes) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=Path("models") / "lstm_experiments")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def slugify(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


if __name__ == "__main__":
    main()
