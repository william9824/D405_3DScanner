import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


LABELS = ["open_palm", "fist", "pinch", "point", "peace"]


class GestureRGBDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        image = Image.open(item["rgb_path"]).convert("RGB")
        label_id = item["label_id"]

        if self.transform:
            image = self.transform(image)

        return image, label_id


class SmallGestureCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 160 -> 80

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 80 -> 40

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 40 -> 20

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 20 -> 10
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 10 * 10, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_samples(dataset_root: Path):
    dataset_root = Path(dataset_root)

    print("\n[Dataset Path Check]")
    print(f"Given dataset_root : {dataset_root}")
    print(f"Absolute path      : {dataset_root.resolve()}")
    print(f"Exists             : {dataset_root.exists()}")

    if dataset_root.exists():
        print("Children:")
        for p in sorted(dataset_root.iterdir()):
            print(" ", p.name)

    samples = []

    for label_id, label in enumerate(LABELS):
        label_dir = dataset_root / label
        print(f"\n[LOAD] label={label}")
        print(f"label_dir: {label_dir}")
        print(f"exists   : {label_dir.exists()}")

        meta_files = sorted(label_dir.glob("*_meta.json"))
        print(f"meta files: {len(meta_files)}")

        for meta_path in meta_files:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            files = meta.get("files", {})
            rgb_path = files.get("rgb_crop")

            # fallback for old format
            if rgb_path is None:
                rgb_path = meta.get("rgb_path")

            if rgb_path is None:
                continue

            rgb_path = Path(rgb_path)

            # Case 1: path stored in metadata works directly
            if rgb_path.exists():
                final_rgb_path = rgb_path

            # Case 2: metadata path was relative to apps/
            elif (Path("apps") / rgb_path).exists():
                final_rgb_path = Path("apps") / rgb_path

            # Case 3: use current label folder + sample_id
            else:
                sample_id = meta.get("sample_id")
                final_rgb_path = label_dir / f"{sample_id}_rgb.png"

            if not final_rgb_path.exists():
                print(f"[WARN] Missing RGB: {final_rgb_path}")
                continue

            samples.append(
                {
                    "rgb_path": final_rgb_path,
                    "label": label,
                    "label_id": label_id,
                    "sample_id": meta.get("sample_id"),
                }
            )

    print(f"\n[LOAD DONE] total samples loaded: {len(samples)}")
    return samples


def split_samples(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    samples = samples.copy()
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    return train_samples, val_samples, test_samples


def count_by_label(samples):
    counts = {label: 0 for label in LABELS}

    for item in samples:
        counts[item["label"]] += 1

    return counts


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    confusion = torch.zeros(len(LABELS), len(LABELS), dtype=torch.int64)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
            confusion[true_label, pred_label] += 1

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, confusion.tolist()


def plot_training_curve(history, out_path: Path):
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
    plt.savefig(out_path.with_name("gesture_rgb_loss_curve.png"), dpi=160)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path.with_name("gesture_rgb_accuracy_curve.png"), dpi=160)
    plt.close()


def plot_confusion_matrix(confusion, labels, out_path: Path):
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
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--dataset-root",
    type=Path,
    default=Path("datasets/gestures"),
    help="Path to gesture dataset root.",
)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("[Gesture RGB Training]")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Device      : {device}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Image size  : {args.image_size}")

    samples = load_samples(args.dataset_root)

    if len(samples) == 0:
        raise RuntimeError("No samples found. Check --dataset-root.")

    print(f"\nTotal samples: {len(samples)}")
    print("All samples label count:")
    print(count_by_label(samples))

    train_samples, val_samples, test_samples = split_samples(
        samples,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=args.seed,
    )

    print("\nSplit:")
    print(f"Train: {len(train_samples)}", count_by_label(train_samples))
    print(f"Val  : {len(val_samples)}", count_by_label(val_samples))
    print(f"Test : {len(test_samples)}", count_by_label(test_samples))

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = GestureRGBDataset(train_samples, transform=train_transform)
    val_dataset = GestureRGBDataset(val_samples, transform=eval_transform)
    test_dataset = GestureRGBDataset(test_samples, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = SmallGestureCNN(num_classes=len(LABELS)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_model_path = args.out_dir / "gesture_rgb_cnn.pt"

    history = []

    print("\nStart training...")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "labels": LABELS,
                "image_size": args.image_size,
                "model_name": "SmallGestureCNN",
                "val_acc": val_acc,
            }

            torch.save(checkpoint, best_model_path)
            print(f"[SAVE] best model -> {best_model_path}")

    print("\nLoading best model for final test...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, confusion = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\n" + "=" * 60)
    print("[Final Test]")
    print("=" * 60)
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc ={test_acc:.4f}")

    print("\nConfusion Matrix")
    print("rows=true, cols=pred")
    print("labels:", LABELS)
    for row in confusion:
        print(row)

    labels_path = args.out_dir / "gesture_rgb_labels.json"
    metrics_path = args.out_dir / "gesture_rgb_metrics.json"

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(LABELS, f, indent=2)

    metrics = {
        "labels": LABELS,
        "dataset_root": str(args.dataset_root).replace("\\", "/"),
        "total_samples": len(samples),
        "split": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "label_count": {
            "all": count_by_label(samples),
            "train": count_by_label(train_samples),
            "val": count_by_label(val_samples),
            "test": count_by_label(test_samples),
        },
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "confusion_matrix": confusion,
        "history": history,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_training_curve(
    history,
    args.out_dir / "gesture_rgb_training_curve.png",
    )

    plot_confusion_matrix(
        confusion,
        LABELS,
        args.out_dir / "gesture_rgb_confusion_matrix.png",
    )

    print(f"[SAVE] loss curve      -> {args.out_dir / 'gesture_rgb_loss_curve.png'}")
    print(f"[SAVE] accuracy curve  -> {args.out_dir / 'gesture_rgb_accuracy_curve.png'}")
    print(f"[SAVE] confusion matrix -> {args.out_dir / 'gesture_rgb_confusion_matrix.png'}")

    print(f"\n[SAVE] labels  -> {labels_path}")
    print(f"[SAVE] metrics -> {metrics_path}")
    print(f"[SAVE] model   -> {best_model_path}")


if __name__ == "__main__":
    main()