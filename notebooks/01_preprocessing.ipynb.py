import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> None:
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=label_names,
        values_format="d",
    )
    plt.tight_layout()
    plt.show()