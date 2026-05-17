#!/usr/bin/env python3
"""
Export long or suspicious dataset texts for manual review, and optionally apply reviewed edits.

Default export format is JSONL. Each row contains:
  - split
  - text_id
  - text_len
  - image_count
  - image_ids
  - text
  - clean_text

Edit clean_text in the review file, then run with --apply-review to create a cleaned dataset dir.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_KEYWORDS = [
    "监控视角",
    "画面",
    "前方骑手",
    "后座乘客",
    "车辆细节",
    "穿搭",
    "姿态",
    "时间戳",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export or apply manual review edits for long MUGE texts.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("cvat_pos_datasets"), help="Input dataset dir.")
    parser.add_argument("--split", default="train", help="Dataset split name, or all for train+valid. Default: train.")
    parser.add_argument("--min-len", type=int, default=120, help="Export texts whose length is >= this value.")
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=DEFAULT_KEYWORDS,
        help="Also export texts containing any of these keywords. Use --keywords with no values to disable.",
    )
    parser.add_argument(
        "--review-out",
        type=Path,
        default=Path("long_texts_review.jsonl"),
        help="Review JSONL path to write.",
    )
    parser.add_argument(
        "--apply-review",
        type=Path,
        default=None,
        help="Reviewed JSONL path. Replaces text with clean_text and writes a new dataset dir.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cvat_pos_datasets_cleaned"),
        help="Output dataset dir when using --apply-review.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output dir.")
    return parser.parse_args()


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


def normalize_review_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def should_export(text: str, min_len: int, keywords: List[str]) -> bool:
    if len(text) >= min_len:
        return True
    return any(keyword and keyword in text for keyword in keywords)


def resolve_splits(split: str) -> List[str]:
    if split == "all":
        return ["train", "valid"]
    return [split]


def export_review(dataset_dir: Path, splits: List[str], min_len: int, keywords: List[str], review_out: Path) -> None:
    total_records = 0
    review_records = []

    for split in splits:
        texts_path = dataset_dir / f"{split}_texts.jsonl"
        if not texts_path.is_file():
            raise FileNotFoundError(f"Missing text file: {texts_path}")

        records = read_jsonl(texts_path)
        total_records += len(records)
        for record in records:
            text = record.get("text", "")
            if not should_export(text, min_len, keywords):
                continue
            image_ids = record.get("image_ids", [])
            review_records.append(
                {
                    "split": split,
                    "text_id": record["text_id"],
                    "text_len": len(text),
                    "image_count": len(image_ids),
                    "image_ids": image_ids,
                    "text": text,
                    "clean_text": text,
                }
            )

    review_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(review_out, review_records)

    print(f"source dataset: {dataset_dir.resolve()}")
    print(f"splits: {', '.join(splits)}")
    print(f"source texts: {total_records} lines")
    print(f"review rows: {len(review_records)}")
    print(f"review file: {review_out.resolve()}")


def load_review_edits(review_path: Path, default_split: str) -> Dict[tuple, str]:
    edits = {}
    for record in read_jsonl(review_path):
        if "text_id" not in record:
            raise ValueError(f"Missing text_id in review row: {record}")
        split = record.get("split", default_split)
        clean_text = normalize_review_text(record.get("clean_text", record.get("text", "")))
        if not clean_text:
            raise ValueError(f"Empty clean_text for split={split} text_id={record['text_id']}")
        edits[(split, int(record["text_id"]))] = clean_text
    return edits


def copy_dataset_files(src_dir: Path, out_dir: Path) -> None:
    for src in sorted(src_dir.iterdir()):
        if src.is_file():
            shutil.copy2(src, out_dir / src.name)


def apply_review(dataset_dir: Path, splits: List[str], review_path: Path, out_dir: Path, overwrite: bool) -> None:
    if not review_path.is_file():
        raise FileNotFoundError(f"Missing review file: {review_path}")
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite to replace files.")

    out_dir.mkdir(parents=True, exist_ok=True)
    copy_dataset_files(dataset_dir, out_dir)

    edits = load_review_edits(review_path, default_split=splits[0])
    changed = 0
    for split in splits:
        texts_path = dataset_dir / f"{split}_texts.jsonl"
        if not texts_path.is_file():
            raise FileNotFoundError(f"Missing text file: {texts_path}")

        records = read_jsonl(texts_path)
        for record in records:
            key = (split, int(record["text_id"]))
            if key not in edits:
                continue
            new_text = edits[key]
            if record.get("text") != new_text:
                changed += 1
                record["text"] = new_text
        write_jsonl(out_dir / f"{split}_texts.jsonl", records)

    print(f"source dataset: {dataset_dir.resolve()}")
    print(f"review file: {review_path.resolve()} ({len(edits)} rows)")
    print(f"splits: {', '.join(splits)}")
    print(f"changed texts: {changed}")
    print(f"output dataset: {out_dir.resolve()}")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    splits = resolve_splits(args.split)

    if args.apply_review is None:
        export_review(
            dataset_dir=dataset_dir,
            splits=splits,
            min_len=args.min_len,
            keywords=args.keywords,
            review_out=args.review_out.expanduser(),
        )
        return

    apply_review(
        dataset_dir=dataset_dir,
        splits=splits,
        review_path=args.apply_review.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
    """
    python3 review_long_texts.py --split all --review-out long_texts_review_all.jsonl
    
    python3 review_long_texts.py --split all --apply-review long_texts_review_all.jsonl --out-dir cvat_pos_datasets_cleaned --overwrite

    
    
    """
