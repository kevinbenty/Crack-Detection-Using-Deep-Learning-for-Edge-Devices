"""Evaluation and reporting utilities shared by all three models.

Kept free of TensorFlow/PyTorch so the Keras scripts and the ViT script
report results in exactly the same way: accuracy, per-class precision /
recall / F1, macro averages (the "Average" rows in Tables V-VII of the
paper) and a confusion matrix.
"""
import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def save_evaluation(y_true, y_pred, class_names, output_dir, title, split="test"):
    """Print and save metrics and a confusion matrix; return them as a dict."""
    os.makedirs(output_dir, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))

    report_text = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        digits=4, zero_division=0,
    )
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    results = {
        "split": split,
        "num_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            name: {k: float(report[name][k]) for k in ("precision", "recall", "f1-score")}
            for name in class_names
        },
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n=== {title}: {split} set ({len(y_true)} images) ===")
    print(report_text)
    print(f"Accuracy: {results['accuracy']:.4f}   Macro F1: {results['macro_f1']:.4f}")

    with open(os.path.join(output_dir, f"{split}_report.txt"), "w") as f:
        f.write(report_text)
    with open(os.path.join(output_dir, f"{split}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=30, values_format="d")
    plt.title(f"{title} - {split} set")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{split}_confusion_matrix.png"), dpi=200)
    plt.close()

    return results


def plot_history(history, output_dir, title):
    """Plot training/validation accuracy and loss curves (paper Fig. 4).

    history is a dict with keys accuracy, val_accuracy, loss, val_loss.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_acc.plot(epochs, history["accuracy"], label="train")
    ax_acc.plot(epochs, history["val_accuracy"], label="validation")
    ax_acc.set(title=f"{title} accuracy", xlabel="epoch", ylabel="accuracy")
    ax_loss.plot(epochs, history["loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.set(title=f"{title} loss", xlabel="epoch", ylabel="loss")
    for ax in (ax_acc, ax_loss):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=200)
    plt.close(fig)


def save_metadata(output_dir, **fields):
    """Save the run configuration next to the results."""
    os.makedirs(output_dir, exist_ok=True)
    fields["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(fields, f, indent=2, default=str)
