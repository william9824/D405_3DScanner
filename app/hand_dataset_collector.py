import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


LABELS = {
    ord("1"): "open_palm",
    ord("2"): "fist",
    ord("3"): "pinch",
    ord("4"): "point",
    ord("5"): "peace",
}

DATASET_ROOT = Path("datasets/gestures")
BURST_COUNT = 30


def ensure_label_folders():
    for label in LABELS.values():
        (DATASET_ROOT / label).mkdir(parents=True, exist_ok=True)


def center_crop(img, crop_size=320):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    half = crop_size // 2

    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w)
    y2 = min(cy + half, h)

    return img[y1:y2, x1:x2], {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "crop_size": int(crop_size),
    }


def save_sample(label, color_img, depth_img, crop_size=320):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    sample_id = f"{timestamp}_{ms:03d}"

    label_dir = DATASET_ROOT / label

    rgb_crop, crop_info = center_crop(color_img, crop_size)
    depth_crop, _ = center_crop(depth_img, crop_size)

    depth_gray = cv2.convertScaleAbs(depth_crop, alpha=0.03)

    rgb_path = label_dir / f"{sample_id}_rgb.png"
    depth_path = label_dir / f"{sample_id}_depth_gray.png"
    meta_path = label_dir / f"{sample_id}_meta.json"

    cv2.imwrite(str(rgb_path), rgb_crop)
    cv2.imwrite(str(depth_path), depth_gray)

    metadata = {
        "sample_id": sample_id,
        "label": label,
        "rgb_path": str(rgb_path),
        "depth_gray_path": str(depth_path),
        "crop": crop_info,
        "created_at": time.time(),
        "camera": "Intel RealSense D405",
        "notes": "center ROI hand gesture sample",
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SAVE] {label}: {sample_id}")


def draw_ui(color_img, depth_gray, current_label):
    depth_color = cv2.applyColorMap(depth_gray, cv2.COLORMAP_JET)

    preview = np.hstack([color_img, depth_color])

    h, w = color_img.shape[:2]
    cx, cy = w // 2, h // 2
    half = 160

    cv2.rectangle(
        preview,
        (cx - half, cy - half),
        (cx + half, cy + half),
        (0, 255, 0),
        2,
    )

    text_lines = [
        f"Current label: {current_label}",
        "1 open_palm | 2 fist | 3 pinch | 4 point | 5 peace",
        "SPACE save one | B burst 30 | Q quit",
    ]

    y = 30
    for line in text_lines:
        cv2.putText(
            preview,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        y += 30

    return preview


def main():
    ensure_label_folders()

    current_label = "open_palm"

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    print("Hand Dataset Collector started.")
    print("1=open_palm, 2=fist, 3=pinch, 4=point, 5=peace")
    print("SPACE=save one, B=burst 30, Q=quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())

            depth_gray = cv2.convertScaleAbs(depth_img, alpha=0.03)

            preview = draw_ui(color_img, depth_gray, current_label)
            cv2.imshow("Hand Dataset Collector - RGB | Depth", preview)

            key = cv2.waitKey(1) & 0xFF

            if key in LABELS:
                current_label = LABELS[key]
                print(f"[LABEL] {current_label}")

            elif key == ord(" "):
                save_sample(current_label, color_img, depth_img)

            elif key in [ord("b"), ord("B")]:
                print(f"[BURST] collecting {BURST_COUNT} samples for {current_label}")
                for i in range(BURST_COUNT):
                    frames = pipeline.wait_for_frames()
                    frames = align.process(frames)

                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                    if not color_frame or not depth_frame:
                        continue

                    color_img = np.asanyarray(color_frame.get_data())
                    depth_img = np.asanyarray(depth_frame.get_data())

                    save_sample(current_label, color_img, depth_img)
                    time.sleep(0.08)

                print("[BURST] done")

            elif key in [ord("q"), ord("Q"), 27]:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()