import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve(history, out_dir: Path):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "gesture_rgb_loss_curve.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "gesture_rgb_accuracy_curve.png", dpi=160)
    plt.close()


def plot_confusion_matrix(confusion, labels, out_dir: Path):
    cm = np.array(confusion)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Gesture RGB Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_dir / "gesture_rgb_confusion_matrix.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("models/gesture_rgb_metrics.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.metrics, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    labels = metrics["labels"]
    history = metrics["history"]
    confusion = metrics["confusion_matrix"]

    plot_training_curve(history, args.out_dir)
    plot_confusion_matrix(confusion, labels, args.out_dir)

    print(f"[SAVE] {args.out_dir / 'gesture_rgb_loss_curve.png'}")
    print(f"[SAVE] {args.out_dir / 'gesture_rgb_accuracy_curve.png'}")
    print(f"[SAVE] {args.out_dir / 'gesture_rgb_confusion_matrix.png'}")


if __name__ == "__main__":
    main()