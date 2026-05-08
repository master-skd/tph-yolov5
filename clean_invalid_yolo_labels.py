#!/usr/bin/env python3
"""Remove unusable YOLO detection labels and their paired images.

This script is intended for detection datasets with this layout:

    dataset_root/
      images/
      labels/

It treats the following labels as invalid:
1. Empty or whitespace-only label files.
2. Any non-empty line that does not have exactly 5 columns.
   This catches segmentation-style labels and malformed annotations.
3. Any line containing non-numeric values.
4. Any class id that is not an integer value.

By default the script runs in dry-run mode and only prints what would be
removed. Pass --apply to actually delete files.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_DATASET_ROOT = Path("/data1/code/xy/datasets/wurenyuan/hengshui/train")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class InvalidLabel:
    label_path: Path
    image_paths: List[Path]
    reason_key: str
    reason_detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete unusable YOLO detection labels and their images."
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=str(DEFAULT_DATASET_ROOT),
        help=f"Dataset root directory (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag the script only previews changes.",
    )
    parser.add_argument(
        "--remove-orphan-images",
        action="store_true",
        help="Also remove images that do not have a matching label file.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=20,
        help="How many sample removals to print (default: 20).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every file that would be removed.",
    )
    return parser.parse_args()


def collect_images(images_dir: Path) -> Dict[str, List[Path]]:
    image_map: Dict[str, List[Path]] = defaultdict(list)
    for image_path in sorted(images_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            image_map[image_path.stem].append(image_path)
    return dict(image_map)


def validate_label(label_path: Path) -> tuple[str | None, str | None]:
    content = label_path.read_text(encoding="utf-8", errors="ignore")
    if not content.strip():
        return "empty_label", "empty or whitespace-only label file"

    found_non_empty_line = False

    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        found_non_empty_line = True
        parts = line.split()
        if len(parts) != 5:
            return (
                "non_detection_label",
                f"line {line_no}: expected 5 columns, got {len(parts)}",
            )

        try:
            class_id = float(parts[0])
            _ = [float(value) for value in parts[1:]]
        except ValueError:
            return (
                "non_numeric_value",
                f"line {line_no}: contains non-numeric values",
            )

        if not class_id.is_integer():
            return (
                "non_integer_class_id",
                f"line {line_no}: class id must be an integer, got {parts[0]}",
            )

    if not found_non_empty_line:
        return "empty_label", "empty or whitespace-only label file"

    return None, None


def find_invalid_labels(
    labels_dir: Path, image_map: Dict[str, List[Path]]
) -> List[InvalidLabel]:
    invalid_labels: List[InvalidLabel] = []

    for label_path in sorted(labels_dir.rglob("*.txt")):
        reason_key, reason_detail = validate_label(label_path)
        if reason_key is None:
            continue

        invalid_labels.append(
            InvalidLabel(
                label_path=label_path,
                image_paths=list(image_map.get(label_path.stem, [])),
                reason_key=reason_key,
                reason_detail=reason_detail or "invalid label",
            )
        )

    return invalid_labels


def find_orphan_images(images_dir: Path, labels_dir: Path) -> List[Path]:
    label_stems = {label_path.stem for label_path in labels_dir.rglob("*.txt")}
    orphan_images: List[Path] = []

    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if image_path.stem not in label_stems:
            orphan_images.append(image_path)

    return orphan_images


def unique_existing_paths(paths: Iterable[Path]) -> List[Path]:
    unique_paths = []
    seen = set()

    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            unique_paths.append(path)

    return unique_paths


def delete_paths(paths: Iterable[Path]) -> int:
    deleted = 0
    for path in unique_existing_paths(paths):
        path.unlink()
        deleted += 1
    return deleted


def print_samples(
    invalid_labels: List[InvalidLabel], orphan_images: List[Path], preview: int, verbose: bool
) -> None:
    if not invalid_labels and not orphan_images:
        return

    if verbose:
        limit = len(invalid_labels) + len(orphan_images)
    else:
        limit = max(preview, 0)

    printed = 0

    for item in invalid_labels:
        if printed >= limit:
            break
        print(f"[label] {item.label_path} | {item.reason_detail}")
        if item.image_paths:
            for image_path in item.image_paths:
                print(f"[image] {image_path}")
        else:
            print("[image] no matching image found")
        printed += 1

    for image_path in orphan_images:
        if printed >= limit:
            break
        print(f"[orphan-image] {image_path}")
        printed += 1

    hidden = len(invalid_labels) + len(orphan_images) - printed
    if hidden > 0:
        print(f"... {hidden} more item(s) not shown. Use --verbose to print all.")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory does not exist: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory does not exist: {labels_dir}")

    image_map = collect_images(images_dir)
    invalid_labels = find_invalid_labels(labels_dir, image_map)
    orphan_images = (
        find_orphan_images(images_dir, labels_dir) if args.remove_orphan_images else []
    )

    label_count = sum(1 for _ in labels_dir.rglob("*.txt"))
    image_count = sum(
        1
        for image_path in images_dir.rglob("*")
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
    )

    reasons = Counter(item.reason_key for item in invalid_labels)
    matched_images_to_remove = unique_existing_paths(
        image_path for item in invalid_labels for image_path in item.image_paths
    )

    print(f"Dataset root: {dataset_root}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Images scanned: {image_count}")
    print(f"Labels scanned: {label_count}")
    print(f"Invalid labels: {len(invalid_labels)}")
    print(f"Matched images to remove: {len(matched_images_to_remove)}")
    if args.remove_orphan_images:
        print(f"Orphan images to remove: {len(orphan_images)}")

    if reasons:
        print("Invalid label reasons:")
        for reason_key, count in sorted(reasons.items()):
            print(f"  - {reason_key}: {count}")

    print_samples(invalid_labels, orphan_images, args.preview, args.verbose)

    if not args.apply:
        print("Dry-run only. Re-run with --apply to delete files.")
        return 0

    deleted_labels = delete_paths(item.label_path for item in invalid_labels)
    deleted_images = delete_paths(matched_images_to_remove)
    deleted_orphans = delete_paths(orphan_images)

    print("Deletion complete.")
    print(f"Deleted labels: {deleted_labels}")
    print(f"Deleted matched images: {deleted_images}")
    if args.remove_orphan_images:
        print(f"Deleted orphan images: {deleted_orphans}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
