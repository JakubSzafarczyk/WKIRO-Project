from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

RANDOM_STATE = 42
DEFAULT_WINDOW_SIZE = 100
DEFAULT_STEP_SIZE = 50
DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)

LABEL_WALK = 0
LABEL_RUN = 1

LABEL_NAMES = {
    LABEL_WALK: "walk",
    LABEL_RUN: "run",
}

ACTIVITY_LABELS = {
    "walk": LABEL_WALK,
    "run": LABEL_RUN,
}

PELVIS_MARKERS = ("LASI", "RASI", "LPSI", "RPSI")

# Real marker coordinates used as the first modelling baseline. The CSV files
# also contain derived angles, forces and segment outputs; those should not be
# translated by a pelvis-centering operation.
DEFAULT_MARKERS = (
    "LFHD",
    "RFHD",
    "LBHD",
    "RBHD",
    "LMAS",
    "RMAS",
    "GLAB",
    "C7",
    "T2",
    "T10",
    "CLAV",
    "STRN",
    "RBAK",
    "LSHO",
    "LUPA",
    "LELB",
    "LFRM",
    "LWRA",
    "LWRB",
    "LFIN",
    "RSHO",
    "RUPA",
    "RELB",
    "RFRM",
    "RWRA",
    "RWRB",
    "RFIN",
    "LASI",
    "RASI",
    "LPSI",
    "RPSI",
    "LTHI",
    "LTHAP",
    "LTHAD",
    "LKNE",
    "LTIB",
    "LTIAP",
    "LTIAD",
    "LANK",
    "LHEE",
    "LTOE",
    "LFMH",
    "LVMH",
    "LFootOff",
    "RTHI",
    "RTHAP",
    "RTHAD",
    "RKNE",
    "RTIB",
    "RTIAP",
    "RTIAD",
    "RANK",
    "RHEE",
    "RTOE",
    "RFMH",
    "RVMH",
    "RFootOff",
)
