import argparse
import json
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


def resolve_sample_paths(label_dir: Path, meta_path: Path, meta: dict):
    sample_id = meta.get("sample_id")
    if sample_id is None:
        sample_id = meta_path.name.replace("_meta.json", "")

    rgb_path = label_dir / f"{sample_id}_rgb.png"
    depth_path = label_dir / f"{sample_id}_depth_gray.png"

    if rgb_path.exists() and depth_path.exists():
        return sample_id, rgb_path, depth_path

    files = meta.get("files", {})
    rgb_fallback = files.get("rgb_crop") or meta.get("rgb_path")
    depth_fallback = files.get("depth_gray_crop") or meta.get("depth_gray_path")

    if rgb_fallback:
        rgb_path = Path(rgb_fallback)
    if depth_fallback:
        depth_path = Path(depth_fallback)

    return sample_id, rgb_path, depth_path


def make_depth_mask(depth_gray: np.ndarray, threshold: int, kernel_size: int):
    """
    depth_gray:
        0 = invalid / background
        > threshold = foreground candidate

    Returns uint8 mask: 0 or 255.
    """
    mask = (depth_gray > threshold).astype(np.uint8) * 255

    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Remove tiny isolated noise.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Fill small holes inside the hand.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Slightly smooth edge.
        mask = cv2.medianBlur(mask, 5)

    return mask


def apply_mask(rgb_bgr: np.ndarray, mask: np.ndarray, background_value: int):
    masked = rgb_bgr.copy()
    masked[mask == 0] = background_value
    return masked


def make_debug_preview(rgb_bgr, depth_gray, mask, masked_bgr):
    depth_bgr = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    rgb_small = cv2.resize(rgb_bgr, (160, 160))
    depth_small = cv2.resize(depth_bgr, (160, 160))
    mask_small = cv2.resize(mask_bgr, (160, 160))
    masked_small = cv2.resize(masked_bgr, (160, 160))

    return np.hstack([rgb_small, depth_small, mask_small, masked_small])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/gestures"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets/gestures_masked_rgb"),
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Depth gray threshold. Pixels <= threshold become background.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Morphology kernel size. Use 1 to disable morphology.",
    )
    parser.add_argument(
        "--background-value",
        type=int,
        default=0,
        help="Background RGB value after masking. 0 = black.",
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=5,
        help="Number of debug preview images per label.",
    )

    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_dataset_root": str(args.dataset_root).replace("\\", "/"),
        "output_dataset_root": str(args.out_root).replace("\\", "/"),
        "threshold": args.threshold,
        "kernel_size": args.kernel_size,
        "background_value": args.background_value,
        "labels": {},
    }

    total_written = 0
    total_missing = 0

    debug_dir = Path("docs/assets/masked_rgb_debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    for label in LABELS:
        src_label_dir = args.dataset_root / label
        dst_label_dir = args.out_root / label
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        meta_files = sorted(src_label_dir.glob("*_meta.json"))

        written = 0
        missing = 0
        mask_valid_ratios = []

        debug_rows = []

        print("\n" + "=" * 60)
        print(f"[LABEL] {label}")
        print("=" * 60)
        print(f"meta files: {len(meta_files)}")

        for meta_path in meta_files:
            meta = load_json(meta_path)
            sample_id, rgb_path, depth_path = resolve_sample_paths(
                src_label_dir,
                meta_path,
                meta,
            )

            if not rgb_path.exists() or not depth_path.exists():
                print(f"[MISSING] {sample_id}")
                print(f"  rgb  : {rgb_path}")
                print(f"  depth: {depth_path}")
                missing += 1
                continue

            rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            depth_gray = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)

            if rgb_bgr is None or depth_gray is None:
                print(f"[READ ERROR] {sample_id}")
                missing += 1
                continue

            mask = make_depth_mask(
                depth_gray=depth_gray,
                threshold=args.threshold,
                kernel_size=args.kernel_size,
            )

            masked_bgr = apply_mask(
                rgb_bgr=rgb_bgr,
                mask=mask,
                background_value=args.background_value,
            )

            mask_valid_ratio = float(np.count_nonzero(mask) / mask.size)
            mask_valid_ratios.append(mask_valid_ratio)

            dst_rgb = dst_label_dir / f"{sample_id}_rgb.png"
            dst_depth = dst_label_dir / f"{sample_id}_depth_gray.png"
            dst_meta = dst_label_dir / f"{sample_id}_meta.json"

            cv2.imwrite(str(dst_rgb), masked_bgr)

            # Keep original depth gray beside masked RGB for future RGB-D experiments.
            shutil.copy2(depth_path, dst_depth)

            new_meta = dict(meta)
            new_meta["preprocess"] = {
                "type": "masked_rgb_from_depth_gray",
                "source_rgb": str(rgb_path).replace("\\", "/"),
                "source_depth_gray": str(depth_path).replace("\\", "/"),
                "threshold": args.threshold,
                "kernel_size": args.kernel_size,
                "background_value": args.background_value,
                "mask_valid_ratio": round(mask_valid_ratio, 4),
            }
            new_meta["files"] = {
                "rgb_crop": str(dst_rgb).replace("\\", "/"),
                "depth_gray_crop": str(dst_depth).replace("\\", "/"),
                "metadata": str(dst_meta).replace("\\", "/"),
            }
            new_meta["notes"] = "Masked RGB dataset generated from D455F depth_gray."

            save_json(dst_meta, new_meta)

            if len(debug_rows) < args.debug_limit:
                debug_rows.append(make_debug_preview(rgb_bgr, depth_gray, mask, masked_bgr))

            written += 1

        if debug_rows:
            debug_grid = np.vstack(debug_rows)
            debug_path = debug_dir / f"{label}_masked_rgb_debug.jpg"
            cv2.imwrite(str(debug_path), debug_grid)
            print(f"[DEBUG] {debug_path}")

        avg_mask_ratio = (
            sum(mask_valid_ratios) / len(mask_valid_ratios)
            if mask_valid_ratios
            else None
        )

        summary["labels"][label] = {
            "source_count": len(meta_files),
            "written": written,
            "missing": missing,
            "avg_mask_valid_ratio": round(avg_mask_ratio, 4)
            if avg_mask_ratio is not None
            else None,
        }

        total_written += written
        total_missing += missing

        print(f"written: {written}")
        print(f"missing: {missing}")
        print(f"avg mask valid ratio: {summary['labels'][label]['avg_mask_valid_ratio']}")

    summary["total_written"] = total_written
    summary["total_missing"] = total_missing

    summary_path = args.out_root / "masked_rgb_summary.json"
    save_json(summary_path, summary)

    print("\n" + "=" * 60)
    print("[DONE]")
    print("=" * 60)
    print(f"total written: {total_written}")
    print(f"total missing: {total_missing}")
    print(f"summary: {summary_path}")
    print(f"debug previews: {debug_dir}")


if __name__ == "__main__":
    main()