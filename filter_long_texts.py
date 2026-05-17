#!/usr/bin/env python3
"""
Remove text records longer than a threshold from a MUGE-style dataset.

For each selected split, the script writes filtered *_texts.jsonl and also filters
the matching *_imgs.tsv so only images still referenced by kept text records remain.
Other files are copied unchanged.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter long texts from a MUGE-style dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("cvat_pos_datasets"), help="Input dataset dir.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cvat_pos_datasets_no_long_texts"),
        help="Output dataset dir.",
    )
    parser.add_argument("--max-len", type=int, default=120, help="Keep texts with len(text) <= this value.")
    parser.add_argument("--split", default="all", help="Split to filter, or all for train+valid. Default: all.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output dir.")
    return parser.parse_args()


def resolve_splits(split: str) -> List[str]:
    if split == "all":
        return ["train", "valid"]
    return [split]


def read_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_tsv(path: Path) -> List[Tuple[int, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid TSV row in {path}:{line_no}")
            rows.append((int(parts[0]), parts[1]))
    return rows


def write_tsv(path: Path, rows: Iterable[Tuple[int, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for image_id, image_b64 in rows:
            f.write(f"{image_id}\t{image_b64}\n")


def referenced_image_ids(records: Iterable[dict]) -> Set[int]:
    ids = set()
    for record in records:
        ids.update(int(image_id) for image_id in record.get("image_ids", []))
    return ids


def copy_dataset_files(src_dir: Path, out_dir: Path) -> None:
    for src in sorted(src_dir.iterdir()):
        if src.is_file():
            shutil.copy2(src, out_dir / src.name)


def filter_split(dataset_dir: Path, out_dir: Path, split: str, max_len: int) -> None:
    texts_path = dataset_dir / f"{split}_texts.jsonl"
    imgs_path = dataset_dir / f"{split}_imgs.tsv"
    if not texts_path.is_file():
        raise FileNotFoundError(f"Missing text file: {texts_path}")
    if not imgs_path.is_file():
        raise FileNotFoundError(f"Missing image file: {imgs_path}")

    records = read_jsonl(texts_path)
    kept_records = [record for record in records if len(record.get("text", "")) <= max_len]
    kept_image_ids = referenced_image_ids(kept_records)

    img_rows = read_tsv(imgs_path)
    kept_img_rows = [(image_id, image_b64) for image_id, image_b64 in img_rows if image_id in kept_image_ids]

    write_jsonl(out_dir / f"{split}_texts.jsonl", kept_records)
    write_tsv(out_dir / f"{split}_imgs.tsv", kept_img_rows)

    removed_texts = len(records) - len(kept_records)
    removed_imgs = len(img_rows) - len(kept_img_rows)
    print(
        f"{split}: texts {len(records)} -> {len(kept_records)} "
        f"(removed {removed_texts}), imgs {len(img_rows)} -> {len(kept_img_rows)} "
        f"(removed {removed_imgs})"
    )


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    splits = resolve_splits(args.split)

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset dir: {dataset_dir}")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite to replace files.")

    out_dir.mkdir(parents=True, exist_ok=True)
    copy_dataset_files(dataset_dir, out_dir)

    for split in splits:
        filter_split(dataset_dir, out_dir, split, args.max_len)

    print(f"output dataset: {out_dir}")


if __name__ == "__main__":
    main()
