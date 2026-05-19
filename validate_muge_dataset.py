#!/usr/bin/env python3
import argparse
import base64
import binascii
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_tsv_rows(path: Path, validate_base64: bool) -> list[tuple[int, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid TSV row in {path}:{line_no}")
            try:
                image_id = int(parts[0])
            except ValueError as exc:
                raise ValueError(f"Invalid image_id in {path}:{line_no}: {parts[0]!r}") from exc
            image_b64 = parts[1]
            if validate_base64:
                try:
                    base64.b64decode(image_b64, validate=True)
                except binascii.Error as exc:
                    raise ValueError(f"Invalid base64 image payload in {path}:{line_no}") from exc
            rows.append((image_id, image_b64))
    return rows


def read_jsonl_records(path: Path) -> tuple[list[dict], list[int]]:
    records = []
    refs = []
    text_ids = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if "text_id" not in item:
                raise ValueError(f"Missing text_id in {path}:{line_no}")
            if not str(item.get("text", "")).strip():
                raise ValueError(f"Empty text in {path}:{line_no} text_id={item.get('text_id')}")
            text_ids.append(int(item["text_id"]))
            records.append(item)
            for image_id in item.get("image_ids", []):
                refs.append(int(image_id))
    duplicate_text_ids = duplicate_values(text_ids)
    if duplicate_text_ids:
        raise ValueError(f"Duplicate text_id in {path}: {duplicate_text_ids[:10]}")
    return records, refs


def duplicate_values(values: list[int]) -> list[int]:
    return [value for value, count in Counter(values).items() if count > 1]


def image_hashes(rows: list[tuple[int, str]]) -> dict[str, int]:
    return {hashlib.sha1(image_b64.encode("utf-8")).hexdigest(): image_id for image_id, image_b64 in rows}


def validate_hard_negatives(
    dataset_dir: Path,
    split: str,
    text_ids: set[int],
    positive_image_ids: set[int],
    validate_base64: bool,
) -> set[int]:
    hard_neg_path = dataset_dir / f"{split}_hard_negatives.jsonl"
    hard_neg_imgs_path = dataset_dir / f"{split}_hard_neg_imgs.tsv"
    if not hard_neg_path.exists() and not hard_neg_imgs_path.exists():
        return set()
    if not hard_neg_path.is_file():
        raise FileNotFoundError(f"Missing hard negative file: {hard_neg_path}")
    if not hard_neg_imgs_path.is_file():
        raise FileNotFoundError(f"Missing hard negative image file: {hard_neg_imgs_path}")

    hard_neg_img_rows = read_tsv_rows(hard_neg_imgs_path, validate_base64=validate_base64)
    hard_neg_img_ids = [image_id for image_id, _ in hard_neg_img_rows]
    duplicate_hard_neg_image_ids = duplicate_values(hard_neg_img_ids)
    if duplicate_hard_neg_image_ids:
        raise ValueError(f"Duplicate image_id in {hard_neg_imgs_path}: {duplicate_hard_neg_image_ids[:10]}")
    hard_neg_img_id_set = set(hard_neg_img_ids)

    records = []
    with hard_neg_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {hard_neg_path}:{line_no}: {exc}") from exc
            records.append(record)

    for record in records:
        text_id = int(record.get("text_id", -1))
        if text_id not in text_ids:
            raise ValueError(f"{hard_neg_path} references missing text_id: {text_id}")

        pos_ids = {int(image_id) for image_id in record.get("positive_image_ids", [])}
        neg_ids = {int(image_id) for image_id in record.get("negative_image_ids", [])}
        missing_pos = sorted(pos_ids - positive_image_ids)
        missing_neg = sorted(neg_ids - hard_neg_img_id_set)
        overlap = sorted(pos_ids & neg_ids)
        if missing_pos:
            raise ValueError(f"{hard_neg_path} text_id={text_id} has missing positive ids: {missing_pos[:10]}")
        if missing_neg:
            raise ValueError(f"{hard_neg_path} text_id={text_id} has missing negative ids: {missing_neg[:10]}")
        if overlap:
            raise ValueError(f"{hard_neg_path} text_id={text_id} uses the same image as positive and negative: {overlap[:10]}")

    print(f"{split}: hard_negative_texts={len(records)}, hard_negative_images={len(hard_neg_img_ids)}, ok")
    return hard_neg_img_id_set


def validate_split(dataset_dir: Path, split: str, require_all_images_used: bool, validate_base64: bool) -> tuple[set[int], set[int], dict[str, int]]:
    texts_path = dataset_dir / f"{split}_texts.jsonl"
    imgs_path = dataset_dir / f"{split}_imgs.tsv"
    if not texts_path.is_file():
        raise FileNotFoundError(f"Missing text file: {texts_path}")
    if not imgs_path.is_file():
        raise FileNotFoundError(f"Missing image file: {imgs_path}")

    img_rows = read_tsv_rows(imgs_path, validate_base64=validate_base64)
    image_ids = [image_id for image_id, _ in img_rows]
    duplicate_image_ids = duplicate_values(image_ids)
    if duplicate_image_ids:
        raise ValueError(f"Duplicate image_id in {imgs_path}: {duplicate_image_ids[:10]}")

    records, refs = read_jsonl_records(texts_path)
    image_id_set = set(image_ids)
    ref_set = set(refs)

    missing = sorted(ref_set - image_id_set)
    if missing:
        raise ValueError(f"{texts_path} references image_ids missing from {imgs_path}: {missing[:10]}")

    unused = sorted(image_id_set - ref_set)
    if require_all_images_used and unused:
        raise ValueError(f"{imgs_path} contains image_ids not referenced by {texts_path}: {unused[:10]}")

    text_ids = {int(record["text_id"]) for record in records}
    validate_hard_negatives(dataset_dir, split, text_ids, image_id_set, validate_base64)

    print(f"{split}: texts={len(records)}, images={len(image_ids)}, refs={len(refs)}, ok")
    return image_id_set, text_ids, image_hashes(img_rows)


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
    parser.add_argument(
        "--allow-cross-split-duplicate-images",
        action="store_true",
        help="Do not fail when different splits contain identical image payloads with different IDs.",
    )
    parser.add_argument(
        "--skip-base64-validation",
        action="store_true",
        help="Skip base64 payload validation for image TSV files.",
    )
    args = parser.parse_args()

    split_image_ids = {}
    split_hashes = {}
    for split in args.splits:
        split_image_ids[split], _, split_hashes[split] = validate_split(
            args.dataset_dir,
            split,
            require_all_images_used=not args.allow_unused_images,
            validate_base64=not args.skip_base64_validation,
        )

    if not args.allow_cross_split_image_ids:
        seen: set[int] = set()
        for split, image_ids in split_image_ids.items():
            overlap = sorted(seen & image_ids)
            if overlap:
                raise ValueError(f"{split} shares image_id with an earlier split: {overlap[:10]}")
            seen.update(image_ids)

    if not args.allow_cross_split_duplicate_images:
        seen_hashes: dict[str, tuple[str, int]] = {}
        for split, hashes in split_hashes.items():
            overlaps = []
            for image_hash, image_id in hashes.items():
                if image_hash in seen_hashes:
                    prev_split, prev_image_id = seen_hashes[image_hash]
                    overlaps.append((prev_split, prev_image_id, split, image_id))
                else:
                    seen_hashes[image_hash] = (split, image_id)
            if overlaps:
                raise ValueError(f"{split} contains images duplicated from earlier splits: {overlaps[:10]}")

    print(f"dataset ok: {args.dataset_dir.resolve()}")


if __name__ == "__main__":
    main()
