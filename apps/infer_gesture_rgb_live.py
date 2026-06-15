import argparse
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


LABELS = ["open_palm", "fist", "pinch", "point", "peace"]

WIDTH = 640
HEIGHT = 480
FPS = 30
CROP_SIZE = 320


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


def center_crop_bgr(img: np.ndarray, crop_size: int):
    h, w = img.shape[:2]
    cx = w // 2
    cy = h // 2
    half = crop_size // 2

    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w)
    y2 = min(cy + half, h)

    crop = img[y1:y2, x1:x2].copy()

    return crop, (x1, y1, x2, y2)


def draw_roi(img: np.ndarray, box):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def draw_prediction(img: np.ndarray, pred_label: str, confidence: float, device):
    if confidence >= 0.75:
        status = "HIGH"
    elif confidence >= 0.50:
        status = "MEDIUM"
    else:
        status = "LOW"

    lines = [
        f"Prediction: {pred_label}",
        f"Confidence: {confidence:.3f} ({status})",
        f"Device: {device}",
        "Q / ESC: quit",
    ]

    y = 35
    for line in lines:
        cv2.putText(
            img,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 32

    if confidence < 0.50:
        cv2.putText(
            img,
            "LOW CONFIDENCE - adjust hand pose / distance",
            (20, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def predict_crop(model, labels, transform, crop_bgr, device):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    x = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    pred_id = int(torch.argmax(probs).item())
    pred_label = labels[pred_id]
    confidence = float(probs[pred_id].item())

    return pred_label, confidence, probs.detach().cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/gesture_rgb_cnn.pt"),
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=CROP_SIZE,
    )
    parser.add_argument(
        "--show-crop",
        action="store_true",
        help="Show cropped ROI window.",
    )
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

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    print("=" * 60)
    print("[D455F Live RGB Gesture Inference]")
    print("=" * 60)
    print(f"Model : {args.model}")
    print(f"Device: {device}")
    print(f"Labels: {labels}")
    print("Keep hand inside the green center ROI.")
    print("Recommended distance: around 0.6m - 1.2m")
    print("Press Q / ESC to quit.")

    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            preview = color_bgr.copy()

            crop_bgr, box = center_crop_bgr(color_bgr, args.crop_size)

            pred_label, confidence, probs = predict_crop(
                model=model,
                labels=labels,
                transform=transform,
                crop_bgr=crop_bgr,
                device=device,
            )

            draw_roi(preview, box)
            draw_prediction(preview, pred_label, confidence, device)

            cv2.imshow("D455F Live Gesture RGB Inference", preview)

            if args.show_crop:
                crop_preview = crop_bgr.copy()
                cv2.putText(
                    crop_preview,
                    f"{pred_label} {confidence:.2f}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Center ROI Crop", crop_preview)

            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()