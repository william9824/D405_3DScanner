import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


WIDTH = 640
HEIGHT = 480
FPS = 30

PROJECT_NAME = "D455F Applied RGB-D Intelligence Lab"
CAMERA_MODEL = "D455F"

DATASET_ROOT = Path("datasets/gestures")
CROP_SIZE = 320
BURST_COUNT = 30
BURST_INTERVAL_S = 0.08

# D455F is not a close-range camera like D405.
# For hand gesture collection, keep the hand roughly around 0.6m - 1.2m.
# This wider valid range gives a little tolerance while still filtering bad depth.
MIN_VALID_DEPTH_M = 0.50
MAX_VALID_DEPTH_M = 1.50

LABELS = {
    ord("1"): "open_palm",
    ord("2"): "fist",
    ord("3"): "pinch",
    ord("4"): "point",
    ord("5"): "peace",
}


def ensure_label_folders(dataset_root: Path) -> None:
    for label in LABELS.values():
        (dataset_root / label).mkdir(parents=True, exist_ok=True)


def make_sample_id() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_%f")[:-3]


def get_device_info(profile) -> dict:
    device = profile.get_device()

    def safe_get(info_key, fallback="Unknown"):
        try:
            return device.get_info(info_key)
        except Exception:
            return fallback

    return {
        "name": safe_get(rs.camera_info.name),
        "serial_number": safe_get(rs.camera_info.serial_number),
        "firmware_version": safe_get(rs.camera_info.firmware_version),
    }


def get_depth_scale(profile) -> float:
    depth_sensor = profile.get_device().first_depth_sensor()
    return float(depth_sensor.get_depth_scale())


def center_crop(img: np.ndarray, crop_size: int = CROP_SIZE):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    half = crop_size // 2

    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w)
    y2 = min(cy + half, h)

    crop = np.ascontiguousarray(img[y1:y2, x1:x2])

    return crop, {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "width": int(x2 - x1),
        "height": int(y2 - y1),
        "crop_size": int(crop_size),
    }


def depth_to_gray(depth_z16: np.ndarray, depth_scale: float) -> np.ndarray:
    """
    Convert z16 depth to 8-bit grayscale for dataset preview/training baseline.

    Invalid depth remains black.
    Nearer valid depth is brighter; farther valid depth is darker.
    """
    depth_m = depth_z16.astype(np.float32) * depth_scale
    depth_clipped = np.clip(depth_m, MIN_VALID_DEPTH_M, MAX_VALID_DEPTH_M)

    depth_norm = (depth_clipped - MIN_VALID_DEPTH_M) / (
        MAX_VALID_DEPTH_M - MIN_VALID_DEPTH_M
    )

    # near = bright, far = dark
    depth_gray = (255.0 - depth_norm * 255.0).astype(np.uint8)

    # invalid depth remains black
    depth_gray[depth_z16 == 0] = 0

    return depth_gray


def compute_depth_stats(depth_z16: np.ndarray, depth_scale: float) -> dict:
    depth_m = depth_z16.astype(np.float32) * depth_scale

    valid_mask = (
        (depth_z16 > 0)
        & (depth_m >= MIN_VALID_DEPTH_M)
        & (depth_m <= MAX_VALID_DEPTH_M)
    )

    total_pixels = int(depth_z16.size)
    valid_pixels = int(np.count_nonzero(valid_mask))
    valid_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0.0

    if valid_pixels == 0:
        return {
            "valid_depth_ratio": 0.0,
            "valid_pixels": valid_pixels,
            "total_pixels": total_pixels,
            "mean_depth_m": None,
            "median_depth_m": None,
            "min_depth_m": None,
            "max_depth_m": None,
            "min_valid_depth_m": MIN_VALID_DEPTH_M,
            "max_valid_depth_m": MAX_VALID_DEPTH_M,
        }

    valid_depth = depth_m[valid_mask]

    return {
        "valid_depth_ratio": round(float(valid_ratio), 4),
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "mean_depth_m": round(float(np.mean(valid_depth)), 4),
        "median_depth_m": round(float(np.median(valid_depth)), 4),
        "min_depth_m": round(float(np.min(valid_depth)), 4),
        "max_depth_m": round(float(np.max(valid_depth)), 4),
        "min_valid_depth_m": MIN_VALID_DEPTH_M,
        "max_valid_depth_m": MAX_VALID_DEPTH_M,
    }


def save_sample(
    label: str,
    color_img: np.ndarray,
    depth_img: np.ndarray,
    depth_scale: float,
    device_info: dict,
    dataset_root: Path,
    crop_size: int,
) -> None:
    sample_id = make_sample_id()
    label_dir = dataset_root / label
    label_dir.mkdir(parents=True, exist_ok=True)

    rgb_crop, crop_info = center_crop(color_img, crop_size)
    depth_crop_z16, _ = center_crop(depth_img, crop_size)

    depth_gray = depth_to_gray(depth_crop_z16, depth_scale)
    depth_stats = compute_depth_stats(depth_crop_z16, depth_scale)

    rgb_path = label_dir / f"{sample_id}_rgb.png"
    depth_path = label_dir / f"{sample_id}_depth_gray.png"
    meta_path = label_dir / f"{sample_id}_meta.json"

    cv2.imwrite(str(rgb_path), rgb_crop)
    cv2.imwrite(str(depth_path), depth_gray)

    metadata = {
        "sample_id": sample_id,
        "project": PROJECT_NAME,
        "capture_type": "hand_gesture_dataset",
        "label": label,
        "camera_model": CAMERA_MODEL,
        "device": device_info,
        "stream": {
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "depth_scale": depth_scale,
        },
        "files": {
            "rgb_crop": str(rgb_path).replace("\\", "/"),
            "depth_gray_crop": str(depth_path).replace("\\", "/"),
            "metadata": str(meta_path).replace("\\", "/"),
        },
        "crop": crop_info,
        "depth": depth_stats,
        "created_at_unix": time.time(),
        "created_at_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "D455F center ROI hand gesture sample",
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"[SAVE] {label}: {sample_id} | "
        f"valid={depth_stats['valid_depth_ratio']} | "
        f"mean_depth={depth_stats['mean_depth_m']}"
    )


def draw_roi_box(img: np.ndarray, crop_size: int) -> None:
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    half = crop_size // 2

    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w)
    y2 = min(cy + half, h)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def draw_ui(
    color_img: np.ndarray,
    depth_img: np.ndarray,
    depth_scale: float,
    current_label: str,
    crop_size: int,
    status: str = "",
) -> np.ndarray:
    rgb_view = color_img.copy()
    depth_gray = depth_to_gray(depth_img, depth_scale)
    depth_view = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)

    draw_roi_box(rgb_view, crop_size)
    draw_roi_box(depth_view, crop_size)

    depth_crop, _ = center_crop(depth_img, crop_size)
    depth_stats = compute_depth_stats(depth_crop, depth_scale)

    preview = np.hstack([rgb_view, depth_view])

    text_lines = [
        f"{PROJECT_NAME} | Current label: {current_label}",
        "1 open_palm | 2 fist | 3 pinch | 4 point | 5 peace",
        "SPACE save one | B burst 30 | Q/ESC quit",
        f"Depth valid: {depth_stats['valid_depth_ratio']:.2f} | Mean depth: {depth_stats['mean_depth_m']}",
        "Recommended hand distance: 0.6m - 1.2m",
    ]

    if status:
        text_lines.append(status)

    y = 30
    for line in text_lines:
        cv2.putText(
            preview,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return preview


def warmup_pipeline(pipeline: rs.pipeline, frame_count: int = 30) -> None:
    for _ in range(frame_count):
        pipeline.wait_for_frames()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect D455F RGB-D hand gesture dataset samples."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Dataset output root. Default: datasets/gestures",
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=CROP_SIZE,
        help="Center crop size in pixels. Default: 320",
    )

    parser.add_argument(
        "--burst-count",
        type=int,
        default=BURST_COUNT,
        help="Number of samples to collect in burst mode. Default: 30",
    )

    parser.add_argument(
        "--burst-interval",
        type=float,
        default=BURST_INTERVAL_S,
        help="Seconds between burst samples. Default: 0.08",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    ensure_label_folders(args.dataset_root)

    current_label = "open_palm"

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_scale = get_depth_scale(profile)
    device_info = get_device_info(profile)

    print("D455F Hand Dataset Collector started.")
    print(f"Device name: {device_info['name']}")
    print(f"Serial     : {device_info['serial_number']}")
    print(f"Depth scale: {depth_scale}")

    if "D455" not in device_info["name"].upper():
        print("[WARN] Connected camera does not look like D455/D455F.")
        print("[WARN] This script is tuned for D455F, not D405 close-range capture.")

    print("1=open_palm, 2=fist, 3=pinch, 4=point, 5=peace")
    print("SPACE=save one, B=burst 30, Q/ESC=quit")
    print(f"Dataset root: {args.dataset_root}")

    warmup_pipeline(pipeline)

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

            preview = draw_ui(
                color_img=color_img,
                depth_img=depth_img,
                depth_scale=depth_scale,
                current_label=current_label,
                crop_size=args.crop_size,
            )

            cv2.imshow("D455F Hand Dataset Collector - RGB | Depth Gray", preview)

            key = cv2.waitKey(1) & 0xFF

            if key in LABELS:
                current_label = LABELS[key]
                print(f"[LABEL] {current_label}")

            elif key == ord(" "):
                save_sample(
                    label=current_label,
                    color_img=color_img,
                    depth_img=depth_img,
                    depth_scale=depth_scale,
                    device_info=device_info,
                    dataset_root=args.dataset_root,
                    crop_size=args.crop_size,
                )

            elif key in [ord("b"), ord("B")]:
                print(
                    f"[BURST] collecting {args.burst_count} samples "
                    f"for {current_label}"
                )

                for i in range(args.burst_count):
                    frames = pipeline.wait_for_frames()
                    frames = align.process(frames)

                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                    if not color_frame or not depth_frame:
                        continue

                    color_img = np.asanyarray(color_frame.get_data())
                    depth_img = np.asanyarray(depth_frame.get_data())

                    status = f"Burst saving {i + 1}/{args.burst_count}"

                    preview = draw_ui(
                        color_img=color_img,
                        depth_img=depth_img,
                        depth_scale=depth_scale,
                        current_label=current_label,
                        crop_size=args.crop_size,
                        status=status,
                    )

                    cv2.imshow(
                        "D455F Hand Dataset Collector - RGB | Depth Gray",
                        preview,
                    )
                    cv2.waitKey(1)

                    save_sample(
                        label=current_label,
                        color_img=color_img,
                        depth_img=depth_img,
                        depth_scale=depth_scale,
                        device_info=device_info,
                        dataset_root=args.dataset_root,
                        crop_size=args.crop_size,
                    )

                    time.sleep(args.burst_interval)

                print("[BURST] done")

            elif key in [ord("q"), ord("Q"), 27]:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()