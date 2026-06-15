import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/gesture_rgb_cnn.pt"))

    args = parser.parse_args()

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

    image = Image.open(args.image).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    pred_id = int(torch.argmax(probs).item())
    pred_label = labels[pred_id]
    confidence = float(probs[pred_id].item())

    print("=" * 50)
    print("[Gesture RGB Image Inference]")
    print("=" * 50)
    print(f"Image      : {args.image}")
    print(f"Device     : {device}")
    print(f"Prediction : {pred_label}")
    print(f"Confidence : {confidence:.4f}")
    print("")
    print("All probabilities:")

    for label, prob in zip(labels, probs.tolist()):
        print(f"  {label:10s}: {prob:.4f}")


if __name__ == "__main__":
    main()