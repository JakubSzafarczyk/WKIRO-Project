from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from .config import DEFAULT_SPLIT_RATIOS, PROJECT_ROOT, RANDOM_STATE
    from .data_loading import TrialRecord
except ImportError:  # pragma: no cover - supports direct imports from notebooks
    from config import DEFAULT_SPLIT_RATIOS, PROJECT_ROOT, RANDOM_STATE
    from data_loading import TrialRecord


SPLIT_NAMES = ("train", "val", "test")


def split_participants(
    participant_ids: Sequence[str],
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    random_state: int = RANDOM_STATE,
) -> dict[str, list[str]]:
    """Create a deterministic train/val/test split over participant IDs."""
    participants = sorted(set(participant_ids))
    if not participants:
        raise ValueError("No participant IDs were provided.")

    counts = _allocate_counts(len(participants), ratios)
    rng = np.random.default_rng(random_state)
    shuffled = np.array(participants, dtype=object)
    rng.shuffle(shuffled)

    result: dict[str, list[str]] = {}
    start = 0
    for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
        stop = start + int(count)
        result[split_name] = sorted(str(item) for item in shuffled[start:stop])
        start = stop

    return result


def create_participant_split(
    records: Sequence[TrialRecord],
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    random_state: int = RANDOM_STATE,
) -> dict[str, list[str]]:
    return split_participants(
        [record.participant_id for record in records],
        ratios=ratios,
        random_state=random_state,
    )


def split_trials_by_participant(
    records: Sequence[TrialRecord],
    participant_splits: dict[str, Sequence[str]],
) -> dict[str, list[TrialRecord]]:
    """Assign every trial to the split of its participant."""
    participant_to_split = {
        participant_id: split_name
        for split_name, participants in participant_splits.items()
        for participant_id in participants
    }

    trial_splits = {split_name: [] for split_name in SPLIT_NAMES}
    missing: set[str] = set()

    for record in records:
        split_name = participant_to_split.get(record.participant_id)
        if split_name is None:
            missing.add(record.participant_id)
            continue
        trial_splits.setdefault(split_name, []).append(record)

    if missing:
        raise ValueError(f"Participants missing from split definition: {sorted(missing)}")

    return trial_splits


def build_split_manifest(
    records: Sequence[TrialRecord],
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    random_state: int = RANDOM_STATE,
    base_dir: Path = PROJECT_ROOT,
) -> dict:
    """Build a JSON-serializable split manifest."""
    participant_splits = create_participant_split(
        records,
        ratios=ratios,
        random_state=random_state,
    )
    trial_splits = split_trials_by_participant(records, participant_splits)

    return {
        "random_state": random_state,
        "ratios": dict(zip(SPLIT_NAMES, ratios, strict=True)),
        "participants": participant_splits,
        "counts": {
            "participants": {
                split_name: len(participants)
                for split_name, participants in participant_splits.items()
            },
            "trials": {
                split_name: len(trials)
                for split_name, trials in trial_splits.items()
            },
        },
        "trials": {
            split_name: [record.to_dict(base_dir=base_dir) for record in trials]
            for split_name, trials in trial_splits.items()
        },
    }


def save_split_manifest(
    records: Sequence[TrialRecord],
    output_path: Path,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    random_state: int = RANDOM_STATE,
    base_dir: Path = PROJECT_ROOT,
) -> dict:
    manifest = build_split_manifest(
        records,
        ratios=ratios,
        random_state=random_state,
        base_dir=base_dir,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _allocate_counts(n_items: int, ratios: Sequence[float]) -> list[int]:
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"Expected {len(SPLIT_NAMES)} split ratios.")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")

    total = float(sum(ratios))
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")

    normalized = np.array(ratios, dtype=float) / total
    raw_counts = normalized * n_items
    counts = np.floor(raw_counts).astype(int)
    remainder = int(n_items - counts.sum())

    if remainder:
        fractional_order = np.argsort(-(raw_counts - counts), kind="stable")
        for index in fractional_order[:remainder]:
            counts[index] += 1

    return counts.tolist()
