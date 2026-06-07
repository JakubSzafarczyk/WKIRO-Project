from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = collect_results(output_dir)

    if not rows:
        raise SystemExit(f"No metrics.json files found under {output_dir}")

    summary = pd.DataFrame(rows).sort_values(
        ["task", "recording_f1_macro", "window_f1_macro"],
        ascending=[True, False, False],
    )
    summary_path = output_dir / "evaluation_summary.csv"
    summary.to_csv(summary_path, index=False)

    best = summary.groupby("task", group_keys=False).head(args.top_k)
    best_path = output_dir / f"best_runs_top{args.top_k}.csv"
    best.to_csv(best_path, index=False)

    print(f"Saved {summary_path}")
    print(f"Saved {best_path}")
    print(best[["task", "run_dir", "recording_accuracy", "recording_f1_macro"]])


def collect_results(output_dir: Path) -> list[dict]:
    rows = []
    for metrics_path in output_dir.rglob("metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        evaluation = payload.get("evaluation")
        if not evaluation:
            continue

        row = {
            "task": payload["task"],
            "run_dir": metrics_path.parent.as_posix(),
            "window_size": payload["window_size"],
            "step_size": payload["step_size"],
            "lstm_units": "x".join(str(value) for value in payload["lstm_units"]),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=Path("models") / "lstm_experiments")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
