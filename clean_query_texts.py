#!/usr/bin/env python3
"""
Apply conservative typo cleanup to MUGE-style query texts.

The script updates:
  - {split}_texts.jsonl
  - {split}_hard_negatives.jsonl, if present

Other dataset files are copied unchanged. Rules are intentionally narrow: they
fix known annotation typos without rewriting valid descriptions such as
"白发老人".
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable


APPAREL_AFTER_WHITE = [
    "上衣",
    "衣服",
    "外套",
    "短袖",
    "长袖",
    "裤子",
    "裤",
    "鞋子",
    "鞋",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean known noisy query text typos in a MUGE dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Input dataset directory.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output cleaned dataset directory.")
    parser.add_argument("--splits", default="train,valid", help="Comma-separated splits to clean.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument("--report-out", type=Path, default=None, help="Optional JSONL report of changed texts.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_dataset_files(src_dir: Path, out_dir: Path) -> None:
    for src in sorted(src_dir.iterdir()):
        if src.is_file():
            shutil.copy2(src, out_dir / src.name)


def clean_text(text: str) -> str:
    text = " ".join(str(text).strip().split())
    replacements = [
        ("戴着眼睛", "戴着眼镜"),
        ("戴眼睛", "戴眼镜"),
        ("黑色眼睛", "黑色眼镜"),
        ("白色眼睛", "白色眼镜"),
        ("学服", "校服"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    apparel_pattern = "|".join(re.escape(term) for term in sorted(APPAREL_AFTER_WHITE, key=len, reverse=True))
    text = re.sub(rf"白发(?=({apparel_pattern}))", "白色", text)
    return text


def clean_records(records: list[dict], split: str, filename: str) -> tuple[list[dict], list[dict]]:
    cleaned = []
    changes = []
    for record in records:
        new_record = dict(record)
        old_text = str(new_record.get("text", ""))
        new_text = clean_text(old_text)
        if new_text != old_text:
            new_record["text"] = new_text
            changes.append(
                {
                    "split": split,
                    "file": filename,
                    "text_id": new_record.get("text_id"),
                    "old_text": old_text,
                    "new_text": new_text,
                }
            )
        cleaned.append(new_record)
    return cleaned, changes


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset dir: {dataset_dir}")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite to replace files.")

    out_dir.mkdir(parents=True, exist_ok=True)
    copy_dataset_files(dataset_dir, out_dir)

    all_changes = []
    for split in splits:
        for suffix in ("texts", "hard_negatives"):
            filename = f"{split}_{suffix}.jsonl"
            path = dataset_dir / filename
            if not path.is_file():
                continue
            records = read_jsonl(path)
            cleaned, changes = clean_records(records, split=split, filename=filename)
            write_jsonl(out_dir / filename, cleaned)
            all_changes.extend(changes)

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.report_out, all_changes)

    print(f"source dataset: {dataset_dir}")
    print(f"output dataset: {out_dir}")
    print(f"changed texts: {len(all_changes)}")
    for change in all_changes[:50]:
        print(f"{change['split']} {change['file']} text_id={change['text_id']}: {change['old_text']} -> {change['new_text']}")
    if len(all_changes) > 50:
        print(f"... {len(all_changes) - 50} more")
    if args.report_out:
        print(f"report: {args.report_out}")


if __name__ == "__main__":
    main()
