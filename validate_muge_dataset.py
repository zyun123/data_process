#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def read_tsv_image_ids(path: Path) -> list[int]:
    image_ids = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid TSV row in {path}:{line_no}")
            try:
                image_ids.append(int(parts[0]))
            except ValueError as exc:
                raise ValueError(f"Invalid image_id in {path}:{line_no}: {parts[0]!r}") from exc
    return image_ids


def read_jsonl_refs(path: Path) -> tuple[int, list[int]]:
    row_count = 0
    refs = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            row_count += 1
            for image_id in item.get("image_ids", []):
                refs.append(int(image_id))
    return row_count, refs


def duplicate_values(values: list[int]) -> list[int]:
    return [value for value, count in Counter(values).items() if count > 1]


def validate_split(dataset_dir: Path, split: str, require_all_images_used: bool) -> set[int]:
    texts_path = dataset_dir / f"{split}_texts.jsonl"
    imgs_path = dataset_dir / f"{split}_imgs.tsv"
    if not texts_path.is_file():
        raise FileNotFoundError(f"Missing text file: {texts_path}")
    if not imgs_path.is_file():
        raise FileNotFoundError(f"Missing image file: {imgs_path}")

    image_ids = read_tsv_image_ids(imgs_path)
    duplicate_image_ids = duplicate_values(image_ids)
    if duplicate_image_ids:
        raise ValueError(f"Duplicate image_id in {imgs_path}: {duplicate_image_ids[:10]}")

    row_count, refs = read_jsonl_refs(texts_path)
    image_id_set = set(image_ids)
    ref_set = set(refs)

    missing = sorted(ref_set - image_id_set)
    if missing:
        raise ValueError(f"{texts_path} references image_ids missing from {imgs_path}: {missing[:10]}")

    unused = sorted(image_id_set - ref_set)
    if require_all_images_used and unused:
        raise ValueError(f"{imgs_path} contains image_ids not referenced by {texts_path}: {unused[:10]}")

    print(f"{split}: texts={row_count}, images={len(image_ids)}, refs={len(refs)}, ok")
    return image_id_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MUGE JSONL image_ids against TSV image IDs.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument(
        "--allow-unused-images",
        action="store_true",
        help="Do not fail when TSV contains images that are not referenced by JSONL.",
    )
    parser.add_argument(
        "--allow-cross-split-image-ids",
        action="store_true",
        help="Do not fail when different splits share the same image_id.",
    )
    args = parser.parse_args()

    split_image_ids = {}
    for split in args.splits:
        split_image_ids[split] = validate_split(
            args.dataset_dir,
            split,
            require_all_images_used=not args.allow_unused_images,
        )

    if not args.allow_cross_split_image_ids:
        seen: set[int] = set()
        for split, image_ids in split_image_ids.items():
            overlap = sorted(seen & image_ids)
            if overlap:
                raise ValueError(f"{split} shares image_id with an earlier split: {overlap[:10]}")
            seen.update(image_ids)

    print(f"dataset ok: {args.dataset_dir.resolve()}")


if __name__ == "__main__":
    main()
