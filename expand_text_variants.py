#!/usr/bin/env python3
"""
Expand MUGE-style text records so one image can match multiple text variants.

The script keeps every original text record, then appends a small number of
rule-based variants that point to the same image_ids. Image TSV files are copied
unchanged because no new images are created.
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


PERSON_TERMS = [
    "人",
    "男子",
    "男人",
    "女士",
    "女人",
    "女子",
    "女孩",
    "男孩",
    "小孩",
    "骑手",
    "乘客",
]
VISUAL_TERMS = [
    "上衣",
    "外套",
    "短袖",
    "长袖",
    "裤",
    "鞋",
    "口罩",
    "帽",
    "头盔",
    "书包",
    "背包",
    "挎包",
    "行李箱",
    "袋子",
    "手机",
    "自行车",
    "电动车",
    "三轮车",
    "车",
]
BACKGROUND_TERMS = [
    "地上",
    "路上",
    "斑马线",
    "网格线",
    "墙",
    "路桩",
    "轿车",
    "监控",
    "画面",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand text variants for a MUGE-style dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("cvat_pos_datasets_no_long_texts"),
        help="Input dataset dir.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cvat_pos_datasets_expanded"),
        help="Output dataset dir.",
    )
    parser.add_argument("--split", default="all", help="Split to expand, or all for train+valid. Default: all.")
    parser.add_argument("--max-new", type=int, default=2, help="Max new variants added for each original text.")
    parser.add_argument("--min-len", type=int, default=8, help="Minimum length for a generated variant.")
    parser.add_argument("--max-len", type=int, default=55, help="Maximum length for a generated variant.")
    parser.add_argument(
        "--single-image-only",
        action="store_true",
        help="Only expand records with exactly one image_id.",
    )
    parser.add_argument(
        "--split-subjects",
        action="store_true",
        help="Allow splitting multi-person texts into separate variants. Off by default because whole-image labels can become noisy.",
    )
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


def copy_dataset_files(src_dir: Path, out_dir: Path) -> None:
    for src in sorted(src_dir.iterdir()):
        if src.is_file():
            shutil.copy2(src, out_dir / src.name)


def normalize_text(text: str) -> str:
    text = " ".join(str(text).strip().split())
    text = text.replace("，。", "。").replace("，，", "，")
    text = text.strip("，。；;、 ")
    return text


def compress_text(text: str) -> str:
    text = normalize_text(text)
    replacements = [
        ("监控视角下，", ""),
        ("画面中，", ""),
        ("画面里，", ""),
        ("一个", ""),
        ("一位", ""),
        ("一名", ""),
        ("穿着", "穿"),
        ("身穿", "穿"),
        ("正在", ""),
        ("正", ""),
        ("的状态", ""),
        ("黑色的", "黑色"),
        ("白色的", "白色"),
        ("蓝色的", "蓝色"),
        ("红色的", "红色"),
        ("黄色的", "黄色"),
        ("绿色的", "绿色"),
        ("灰色的", "灰色"),
        ("棕色的", "棕色"),
        ("紫色的", "紫色"),
        ("粉色的", "粉色"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    text = text.replace("在路上行走", "行走").replace("行走在路上", "行走")
    return normalize_text(text)


def has_visual_value(text: str) -> bool:
    return any(term in text for term in VISUAL_TERMS) and any(term in text for term in PERSON_TERMS + ["车"])


def remove_background_clauses(text: str) -> str:
    clauses = [normalize_text(part) for part in re.split(r"[，,]", text) if normalize_text(part)]
    kept = []
    for clause in clauses:
        if any(term in clause for term in BACKGROUND_TERMS) and not any(term in clause for term in PERSON_TERMS):
            continue
        kept.append(clause)
    return normalize_text("，".join(kept))


def split_subjects(text: str) -> List[str]:
    parts = re.split(r"[；;。]|，(?=旁边有|后面有|前面有|左侧|右侧|另一个|还有)", text)
    variants = []
    for part in parts:
        part = normalize_text(part)
        part = re.sub(r"^(旁边有|后面有|前面有|还有)", "", part)
        part = compress_text(part)
        if part:
            variants.append(part)
    return variants


def add_candidate(candidates: List[str], text: str, original: str, min_len: int, max_len: int) -> None:
    text = normalize_text(text)
    if not text or text == original:
        return
    if len(text) < min_len or len(text) > max_len:
        return
    if not has_visual_value(text):
        return
    if text not in candidates:
        candidates.append(text)


def generate_variants(text: str, min_len: int, max_len: int, max_new: int, split_subjects_enabled: bool) -> List[str]:
    original = normalize_text(text)
    candidates: List[str] = []

    compact = compress_text(original)
    add_candidate(candidates, compact, original, min_len, max_len)

    no_background = remove_background_clauses(compact)
    add_candidate(candidates, no_background, original, min_len, max_len)

    if split_subjects_enabled:
        for part in split_subjects(original):
            add_candidate(candidates, part, original, min_len, max_len)

    return candidates[:max_new]


def next_text_id(records: Sequence[dict]) -> int:
    if not records:
        return 0
    return max(int(record["text_id"]) for record in records) + 1


def expand_split(
    dataset_dir: Path,
    out_dir: Path,
    split: str,
    max_new: int,
    min_len: int,
    max_len: int,
    single_image_only: bool,
    split_subjects_enabled: bool,
) -> None:
    texts_path = dataset_dir / f"{split}_texts.jsonl"
    if not texts_path.is_file():
        raise FileNotFoundError(f"Missing text file: {texts_path}")

    records = read_jsonl(texts_path)
    expanded = [dict(record) for record in records]
    next_id = next_text_id(records)
    seen = {(normalize_text(record.get("text", "")), tuple(record.get("image_ids", []))) for record in records}
    added = 0

    for record in records:
        image_ids = record.get("image_ids", [])
        if single_image_only and len(image_ids) != 1:
            continue
        variants = generate_variants(
            record.get("text", ""),
            min_len=min_len,
            max_len=max_len,
            max_new=max_new,
            split_subjects_enabled=split_subjects_enabled,
        )
        for variant in variants:
            key = (variant, tuple(image_ids))
            if key in seen:
                continue
            expanded.append(
                {
                    "text_id": next_id,
                    "text": variant,
                    "image_ids": image_ids,
                }
            )
            seen.add(key)
            next_id += 1
            added += 1

    write_jsonl(out_dir / f"{split}_texts.jsonl", expanded)
    print(f"{split}: texts {len(records)} -> {len(expanded)} (added {added})")


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
        expand_split(
            dataset_dir=dataset_dir,
            out_dir=out_dir,
            split=split,
            max_new=args.max_new,
            min_len=args.min_len,
            max_len=args.max_len,
            single_image_only=args.single_image_only,
            split_subjects_enabled=args.split_subjects,
        )

    print(f"output dataset: {out_dir}")


if __name__ == "__main__":
    main()
