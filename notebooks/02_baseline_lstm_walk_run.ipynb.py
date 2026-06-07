from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR
from models import build_lstm_classifier
from evaluation import evaluate_predictions, plot_confusion_matrix

np.random.seed(42)
tf.random.set_seed(42)