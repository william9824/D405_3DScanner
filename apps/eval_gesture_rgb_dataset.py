import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


LABELS = ["open_palm", "fist", "pinch", "point", "peace"]


class SmallGestureCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
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


def load_model(model_path: Path, device):
    checkpoint = torch.load(model_path, map_location=device)

    labels = checkpoint.get("labels", LABELS)
    image_size = checkpoint.get("image_size", 160)

    model = SmallGestureCNN(num_classes=len(labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, labels, image_size


def load_samples(dataset_root: Path, labels):
    samples = []

    for label in labels:
        label_dir = dataset_root / label
        meta_files = sorted(label_dir.glob("*_meta.json"))

        for meta_path in meta_files:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            sample_id = meta["sample_id"]
            rgb_path = label_dir / f"{sample_id}_rgb.png"

            if not rgb_path.exists():
                files = meta.get("files", {})
                rgb_path = Path(files.get("rgb_crop", ""))

            if not rgb_path.exists():
                print(f"[WARN] Missing RGB: {meta_path}")
                continue

            samples.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "rgb_path": str(rgb_path),
                    "meta_path": str(meta_path),
                }
            )

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/gestures"))
    parser.add_argument("--model", type=Path, default=Path("models/gesture_rgb_cnn.pt"))
    parser.add_argument("--out", type=Path, default=Path("reports/gesture_rgb_eval_report.json"))
    parser.add_argument("--low-confidence-threshold", type=float, default=0.60)

    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, labels, image_size = load_model(args.model, device)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    samples = load_samples(args.dataset_root, labels)

    print("=" * 60)
    print("[Gesture RGB Dataset Evaluation]")
    print("=" * 60)
    print(f"Dataset root : {args.dataset_root}")
    print(f"Model        : {args.model}")
    print(f"Device       : {device}")
    print(f"Samples      : {len(samples)}")
    print("")

    total = 0
    correct = 0

    per_class = {
        label: {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
        }
        for label in labels
    }

    wrong_samples = []
    low_confidence_samples = []

    confusion = {
        true_label: {pred_label: 0 for pred_label in labels}
        for true_label in labels
    }

    with torch.no_grad():
        for sample in samples:
            image = Image.open(sample["rgb_path"]).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]

            pred_id = int(torch.argmax(probs).item())
            pred_label = labels[pred_id]
            confidence = float(probs[pred_id].item())

            true_label = sample["label"]

            total += 1
            per_class[true_label]["total"] += 1
            confusion[true_label][pred_label] += 1

            is_correct = pred_label == true_label

            if is_correct:
                correct += 1
                per_class[true_label]["correct"] += 1
            else:
                wrong_samples.append(
                    {
                        **sample,
                        "prediction": pred_label,
                        "confidence": confidence,
                        "probabilities": {
                            label: float(prob)
                            for label, prob in zip(labels, probs.tolist())
                        },
                    }
                )

            if confidence < args.low_confidence_threshold:
                low_confidence_samples.append(
                    {
                        **sample,
                        "prediction": pred_label,
                        "confidence": confidence,
                        "correct": is_correct,
                        "probabilities": {
                            label: float(prob)
                            for label, prob in zip(labels, probs.tolist())
                        },
                    }
                )

    overall_acc = correct / total if total > 0 else 0.0

    for label in labels:
        c = per_class[label]["correct"]
        t = per_class[label]["total"]
        per_class[label]["accuracy"] = c / t if t > 0 else 0.0

    report = {
        "dataset_root": str(args.dataset_root).replace("\\", "/"),
        "model": str(args.model).replace("\\", "/"),
        "device": str(device),
        "total": total,
        "correct": correct,
        "overall_accuracy": overall_acc,
        "per_class": per_class,
        "confusion": confusion,
        "wrong_count": len(wrong_samples),
        "low_confidence_threshold": args.low_confidence_threshold,
        "low_confidence_count": len(low_confidence_samples),
        "wrong_samples": wrong_samples,
        "low_confidence_samples": low_confidence_samples[:100],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Overall accuracy: {overall_acc:.4f}")
    print("")
    print("Per-class accuracy:")

    for label in labels:
        info = per_class[label]
        print(
            f"  {label:10s}: "
            f"{info['accuracy']:.4f} "
            f"({info['correct']}/{info['total']})"
        )

    print("")
    print(f"Wrong samples         : {len(wrong_samples)}")
    print(f"Low confidence samples: {len(low_confidence_samples)}")
    print(f"[SAVE] {args.out}")

    if wrong_samples:
        print("")
        print("First 10 wrong samples:")
        for item in wrong_samples[:10]:
            print(
                f"  true={item['label']:10s} "
                f"pred={item['prediction']:10s} "
                f"conf={item['confidence']:.4f} "
                f"path={item['rgb_path']}"
            )


if __name__ == "__main__":
    main()