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


def make_depth_mask(depth_gray: np.ndarray, threshold: int):
    return (depth_gray > threshold).astype(np.uint8) * 255


def make_skin_mask_ycrcb(rgb_bgr: np.ndarray):
    """
    Simple skin-color mask in YCrCb space.
    Works reasonably well for quick hand/arm segmentation under normal lighting.
    """
    ycrcb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2YCrCb)

    # OpenCV YCrCb channel order = Y, Cr, Cb
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)

    skin_mask = cv2.inRange(ycrcb, lower, upper)
    return skin_mask


def clean_mask(mask: np.ndarray, kernel_size: int, min_area: int, keep_components: int):
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.medianBlur(mask, 5)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    components = []

    for component_id in range(1, num_labels):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        if w < 10 or h < 10:
            continue

        components.append(
            {
                "id": int(component_id),
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )

    components = sorted(components, key=lambda item: item["area"], reverse=True)
    components = components[:keep_components]

    final_mask = np.zeros_like(mask)

    for comp in components:
        final_mask[labels == comp["id"]] = 255

    return final_mask, components


def apply_mask(rgb_bgr: np.ndarray, mask: np.ndarray, background_value: int):
    masked = rgb_bgr.copy()
    masked[mask == 0] = background_value
    return masked


def make_debug_preview(rgb_bgr, depth_gray, depth_mask, skin_mask, final_mask, masked_bgr):
    rgb_small = cv2.resize(rgb_bgr, (160, 160))

    depth_bgr = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
    depth_small = cv2.resize(depth_bgr, (160, 160))

    depth_mask_bgr = cv2.cvtColor(depth_mask, cv2.COLOR_GRAY2BGR)
    depth_mask_small = cv2.resize(depth_mask_bgr, (160, 160))

    skin_mask_bgr = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
    skin_mask_small = cv2.resize(skin_mask_bgr, (160, 160))

    final_mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    final_mask_small = cv2.resize(final_mask_bgr, (160, 160))

    masked_small = cv2.resize(masked_bgr, (160, 160))

    return np.hstack([
        rgb_small,
        depth_small,
        depth_mask_small,
        skin_mask_small,
        final_mask_small,
        masked_small,
    ])


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
        default=Path("datasets/gestures_hand_masked_rgb"),
    )
    parser.add_argument(
        "--depth-threshold",
        type=int,
        default=20,
        help="Depth gray threshold. Higher = stricter foreground.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Small connected components below this area are removed.",
    )
    parser.add_argument(
        "--keep-components",
        type=int,
        default=2,
        help="Keep largest N skin-depth components. Hand + arm may be separated.",
    )
    parser.add_argument(
        "--background-value",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    debug_dir = Path("docs/assets/hand_masked_rgb_debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_dataset_root": str(args.dataset_root).replace("\\", "/"),
        "output_dataset_root": str(args.out_root).replace("\\", "/"),
        "method": "depth_mask AND YCrCb skin_mask",
        "depth_threshold": args.depth_threshold,
        "kernel_size": args.kernel_size,
        "min_area": args.min_area,
        "keep_components": args.keep_components,
        "labels": {},
    }

    total_written = 0
    total_missing = 0

    for label in LABELS:
        src_label_dir = args.dataset_root / label
        dst_label_dir = args.out_root / label
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        meta_files = sorted(src_label_dir.glob("*_meta.json"))

        written = 0
        missing = 0
        mask_ratios = []
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
                missing += 1
                continue

            rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            depth_gray = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)

            if rgb_bgr is None or depth_gray is None:
                print(f"[READ ERROR] {sample_id}")
                missing += 1
                continue

            depth_mask = make_depth_mask(depth_gray, args.depth_threshold)
            skin_mask = make_skin_mask_ycrcb(rgb_bgr)

            combined_mask = cv2.bitwise_and(depth_mask, skin_mask)

            final_mask, components = clean_mask(
                combined_mask,
                kernel_size=args.kernel_size,
                min_area=args.min_area,
                keep_components=args.keep_components,
            )

            masked_bgr = apply_mask(
                rgb_bgr=rgb_bgr,
                mask=final_mask,
                background_value=args.background_value,
            )

            mask_valid_ratio = float(np.count_nonzero(final_mask) / final_mask.size)
            mask_ratios.append(mask_valid_ratio)

            dst_rgb = dst_label_dir / f"{sample_id}_rgb.png"
            dst_depth = dst_label_dir / f"{sample_id}_depth_gray.png"
            dst_meta = dst_label_dir / f"{sample_id}_meta.json"

            cv2.imwrite(str(dst_rgb), masked_bgr)
            shutil.copy2(depth_path, dst_depth)

            new_meta = dict(meta)
            new_meta["preprocess"] = {
                "type": "hand_masked_rgb_from_depth_and_skin",
                "source_rgb": str(rgb_path).replace("\\", "/"),
                "source_depth_gray": str(depth_path).replace("\\", "/"),
                "depth_threshold": args.depth_threshold,
                "kernel_size": args.kernel_size,
                "min_area": args.min_area,
                "keep_components": args.keep_components,
                "mask_valid_ratio": round(mask_valid_ratio, 4),
                "components": components,
            }

            new_meta["files"] = {
                "rgb_crop": str(dst_rgb).replace("\\", "/"),
                "depth_gray_crop": str(dst_depth).replace("\\", "/"),
                "metadata": str(dst_meta).replace("\\", "/"),
            }

            new_meta["notes"] = "Hand-masked RGB generated using depth + YCrCb skin mask."

            save_json(dst_meta, new_meta)

            if len(debug_rows) < args.debug_limit:
                debug_rows.append(
                    make_debug_preview(
                        rgb_bgr,
                        depth_gray,
                        depth_mask,
                        skin_mask,
                        final_mask,
                        masked_bgr,
                    )
                )

            written += 1

        if debug_rows:
            debug_grid = np.vstack(debug_rows)
            debug_path = debug_dir / f"{label}_hand_masked_rgb_debug.jpg"
            cv2.imwrite(str(debug_path), debug_grid)
            print(f"[DEBUG] {debug_path}")

        avg_mask_ratio = sum(mask_ratios) / len(mask_ratios) if mask_ratios else None

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

    summary_path = args.out_root / "hand_masked_rgb_summary.json"
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