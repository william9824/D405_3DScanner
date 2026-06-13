import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


LABELS = ["open_palm", "fist", "pinch", "point", "peace"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def sanitize_metadata(data: dict, dst_rgb: Path, dst_depth: Path, dst_meta: Path):
    public_meta = dict(data)

    device = dict(public_meta.get("device", {}))
    if "serial_number" in device:
        device["serial_number"] = "redacted_for_public_demo"
    public_meta["device"] = device

    public_meta["files"] = {
        "rgb_crop": str(dst_rgb).replace("\\", "/"),
        "depth_gray_crop": str(dst_depth).replace("\\", "/"),
        "metadata": str(dst_meta).replace("\\", "/"),
    }

    public_meta["public_subset"] = True
    public_meta["notes"] = "Public demo subset sample. Full dataset is kept local."

    return public_meta


def find_sample_files(label_dir: Path, meta_path: Path, data: dict):
    sample_id = data.get("sample_id")

    if sample_id is None:
        sample_id = meta_path.name.replace("_meta.json", "")

    rgb_path = label_dir / f"{sample_id}_rgb.png"
    depth_path = label_dir / f"{sample_id}_depth_gray.png"

    if rgb_path.exists() and depth_path.exists():
        return sample_id, rgb_path, depth_path

    files = data.get("files", {})
    rgb_fallback = files.get("rgb_crop") or data.get("rgb_path")
    depth_fallback = files.get("depth_gray_crop") or data.get("depth_gray_path")

    if rgb_fallback:
        rgb_path = Path(rgb_fallback)
    if depth_fallback:
        depth_path = Path(depth_fallback)

    return sample_id, rgb_path, depth_path


def avg(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def make_preview_grid(out_root: Path, preview_path: Path, max_per_label: int = 2):
    rows = []

    for label in LABELS:
        label_dir = out_root / label
        rgb_files = sorted(label_dir.glob("*_rgb.png"))[:max_per_label]

        cells = []

        title = np.zeros((160, 140, 3), dtype=np.uint8)
        cv2.putText(
            title,
            label,
            (8, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cells.append(title)

        for rgb_path in rgb_files:
            sample_id = rgb_path.name.replace("_rgb.png", "")
            depth_path = label_dir / f"{sample_id}_depth_gray.png"

            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)

            if rgb is None or depth is None:
                continue

            rgb = cv2.resize(rgb, (160, 160))
            depth = cv2.resize(depth, (160, 160))
            depth_bgr = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

            cells.append(rgb)
            cells.append(depth_bgr)

        while len(cells) < 1 + max_per_label * 2:
            cells.append(np.zeros((160, 160, 3), dtype=np.uint8))

        rows.append(np.hstack(cells))

    grid = np.vstack(rows)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), grid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("apps/datasets/gestures"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets_sample/gestures"),
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    args.out_root.mkdir(parents=True, exist_ok=True)

    full_summary = {
        "project": "D455F Applied RGB-D Intelligence Lab",
        "full_dataset_root": str(args.dataset_root).replace("\\", "/"),
        "public_subset_root": str(args.out_root).replace("\\", "/"),
        "samples_per_label": args.samples_per_label,
        "labels": {},
    }

    for label in LABELS:
        src_label_dir = args.dataset_root / label
        dst_label_dir = args.out_root / label
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        meta_files = sorted(src_label_dir.glob("*_meta.json"))

        if len(meta_files) == 0:
            print(f"[WARN] No metadata found for label: {label}")
            continue

        selected = meta_files.copy()
        rng.shuffle(selected)
        selected = selected[: args.samples_per_label]

        valid_ratios = []
        mean_depths = []
        copied = 0
        missing = 0

        for meta_path in selected:
            data = load_json(meta_path)
            sample_id, rgb_path, depth_path = find_sample_files(src_label_dir, meta_path, data)

            if not rgb_path.exists() or not depth_path.exists():
                print(f"[MISSING] {label} {sample_id}")
                missing += 1
                continue

            dst_rgb = dst_label_dir / f"{sample_id}_rgb.png"
            dst_depth = dst_label_dir / f"{sample_id}_depth_gray.png"
            dst_meta = dst_label_dir / f"{sample_id}_meta.json"

            shutil.copy2(rgb_path, dst_rgb)
            shutil.copy2(depth_path, dst_depth)

            public_meta = sanitize_metadata(data, dst_rgb, dst_depth, dst_meta)
            save_json(dst_meta, public_meta)

            depth = data.get("depth", {})
            valid_ratios.append(depth.get("valid_depth_ratio"))
            mean_depths.append(depth.get("mean_depth_m"))

            copied += 1

        all_valid_ratios = []
        all_mean_depths = []

        for meta_path in meta_files:
            data = load_json(meta_path)
            depth = data.get("depth", {})
            all_valid_ratios.append(depth.get("valid_depth_ratio"))
            all_mean_depths.append(depth.get("mean_depth_m"))

        full_summary["labels"][label] = {
            "full_count": len(meta_files),
            "public_subset_count": copied,
            "missing_during_export": missing,
            "avg_valid_depth_ratio_full": avg(all_valid_ratios),
            "avg_mean_depth_m_full": avg(all_mean_depths),
            "avg_valid_depth_ratio_subset": avg(valid_ratios),
            "avg_mean_depth_m_subset": avg(mean_depths),
        }

        print(f"[OK] {label}: copied {copied}/{len(meta_files)}")

    total_full = sum(v["full_count"] for v in full_summary["labels"].values())
    total_subset = sum(v["public_subset_count"] for v in full_summary["labels"].values())

    full_summary["total_full_samples"] = total_full
    full_summary["total_public_subset_samples"] = total_subset

    save_json(Path("dataset_summary.json"), full_summary)

    make_preview_grid(
        out_root=args.out_root,
        preview_path=Path("docs/assets/gesture_sample_grid.jpg"),
        max_per_label=2,
    )

    print("\n[DONE]")
    print(f"Full samples   : {total_full}")
    print(f"Public samples : {total_subset}")
    print("Wrote: dataset_summary.json")
    print("Wrote: docs/assets/gesture_sample_grid.jpg")


if __name__ == "__main__":
    main()