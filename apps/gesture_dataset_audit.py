import json
from pathlib import Path
from collections import defaultdict

import cv2


DATASET_ROOT = Path("datasets/gestures")
LABELS = ["open_palm", "fist", "pinch", "point", "peace"]


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return None, str(e)


def check_image(path: Path):
    if not path.exists():
        return False, "missing"

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False, "cannot_read"

    return True, img.shape


def main():
    summary = {}

    total_samples = 0
    total_bad = 0

    for label in LABELS:
        label_dir = DATASET_ROOT / label
        meta_files = sorted(label_dir.glob("*_meta.json"))

        count = 0
        d455f_count = 0
        old_format_count = 0
        missing_file_count = 0
        low_depth_count = 0

        valid_ratios = []
        mean_depths = []

        print("\n" + "=" * 60)
        print(f"[LABEL] {label}")
        print("=" * 60)

        for meta_path in meta_files:
            data = load_json(meta_path)

            if isinstance(data, tuple):
                print(f"[BAD JSON] {meta_path}: {data[1]}")
                total_bad += 1
                continue

            count += 1
            total_samples += 1

            camera_model = data.get("camera_model")
            if camera_model == "D455F":
                d455f_count += 1
            else:
                old_format_count += 1
                print(f"[OLD/UNKNOWN META] {meta_path}")

            files = data.get("files", {})
            rgb_path = files.get("rgb_crop")
            depth_path = files.get("depth_gray_crop")

            # fallback for old metadata format
            if rgb_path is None:
                rgb_path = data.get("rgb_path")
            if depth_path is None:
                depth_path = data.get("depth_gray_path")

            if rgb_path is None or depth_path is None:
                print(f"[MISSING PATH FIELD] {meta_path}")
                missing_file_count += 1
                total_bad += 1
                continue

            rgb_ok, rgb_info = check_image(Path(rgb_path))
            depth_ok, depth_info = check_image(Path(depth_path))

            if not rgb_ok or not depth_ok:
                print(f"[MISSING IMAGE] {meta_path}")
                print(f"  rgb   : {rgb_path} -> {rgb_info}")
                print(f"  depth : {depth_path} -> {depth_info}")
                missing_file_count += 1
                total_bad += 1
                continue

            depth = data.get("depth", {})
            valid_ratio = depth.get("valid_depth_ratio")
            mean_depth = depth.get("mean_depth_m")

            if valid_ratio is not None:
                valid_ratios.append(valid_ratio)

                if valid_ratio < 0.10:
                    low_depth_count += 1
                    print(
                        f"[LOW DEPTH] {meta_path.name} "
                        f"valid_depth_ratio={valid_ratio}"
                    )

            if mean_depth is not None:
                mean_depths.append(mean_depth)

        avg_valid = sum(valid_ratios) / len(valid_ratios) if valid_ratios else None
        avg_depth = sum(mean_depths) / len(mean_depths) if mean_depths else None

        summary[label] = {
            "total_meta": count,
            "d455f_meta": d455f_count,
            "old_or_unknown_meta": old_format_count,
            "missing_file_count": missing_file_count,
            "low_depth_count": low_depth_count,
            "avg_valid_depth_ratio": avg_valid,
            "avg_mean_depth_m": avg_depth,
        }

        print(f"total meta          : {count}")
        print(f"D455F meta          : {d455f_count}")
        print(f"old/unknown meta    : {old_format_count}")
        print(f"missing files       : {missing_file_count}")
        print(f"low depth < 0.10    : {low_depth_count}")

        if avg_valid is not None:
            print(f"avg valid ratio     : {avg_valid:.4f}")
        else:
            print("avg valid ratio     : None")

        if avg_depth is not None:
            print(f"avg mean depth      : {avg_depth:.4f} m")
        else:
            print("avg mean depth      : None")

    print("\n" + "=" * 60)
    print("[DATASET SUMMARY]")
    print("=" * 60)

    for label, info in summary.items():
        print(
            f"{label:10s} | "
            f"total={info['total_meta']:4d} | "
            f"D455F={info['d455f_meta']:4d} | "
            f"old={info['old_or_unknown_meta']:4d} | "
            f"missing={info['missing_file_count']:3d} | "
            f"low_depth={info['low_depth_count']:3d} | "
            f"avg_valid={info['avg_valid_depth_ratio']}"
        )

    print("\nTotal samples:", total_samples)
    print("Total bad    :", total_bad)


if __name__ == "__main__":
    main()