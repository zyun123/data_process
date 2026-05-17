#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expand Chinese-CLIP train_texts.jsonl for surveillance attribute retrieval.

Input line format:
  {"text_id": 0, "text": "...", "image_ids": [1,2,3]}

Output keeps original samples and adds derived positive queries that are
substring/regex-based from the original text, avoiding most background phrases.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from collections import Counter, defaultdict

COLORS = ["黑色", "白色", "蓝色", "红色", "黄色", "灰色", "粉色", "绿色", "紫色", "棕色", "橙色", "银色", "金色"]
# also match short colors in text like 黑衣服/白裤子
SHORT_COLOR_MAP = {
    "黑": "黑色", "白": "白色", "蓝": "蓝色", "红": "红色", "黄": "黄色", "灰": "灰色",
    "粉": "粉色", "绿": "绿色", "紫": "紫色", "棕": "棕色", "橙": "橙色",
}

PERSON_WORDS = ["男子", "男人", "男孩", "女人", "女子", "女孩", "老人", "小孩", "儿童", "行人", "人"]

# Background/noise clauses are useful for captioning, but usually harmful for person attribute retrieval.
NOISE_PATTERNS = [
    r"地上有[^，。；;]*", r"路上有[^，。；;]*", r"旁边有[^，。；;]*", r"背景[^，。；;]*",
    r"行走在[^，。；;]*", r"走在[^，。；;]*", r"站在[^，。；;]*", r"位于[^，。；;]*",
    r"斑马线", r"网格路", r"道路上", r"马路上",
]

# Prefer these object words for retrieval attributes.
ATTR_OBJECTS = [
    "短袖", "长袖", "上衣", "衣服", "外套", "雨衣", "裤子", "短裤", "鞋子", "鞋",
    "头盔", "帽子", "口罩", "书包", "背包", "包", "袋子", "手提袋",
    "电动车", "自行车", "摩托车", "挡风被",
]

# Object group order matters for combination queries.
GROUP_ORDER = ["mask", "helmet", "hat", "upper", "pants", "shoes", "bag", "vehicle", "windshield", "action"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e
    return rows


def normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("，", "，").replace("。", "")
    s = re.sub(r"\s+", "", s)
    # Normalize common redundant wording.
    s = s.replace("穿着", "穿")
    s = s.replace("带着", "戴")
    s = s.replace("背着", "背")
    s = s.replace("骑着", "骑")
    s = s.replace("的的", "的")
    return s.strip("，；;。 ")


def clean_background(s: str) -> str:
    s = normalize_text(s)
    # Remove noise substrings inside clauses.
    for pat in NOISE_PATTERNS:
        s = re.sub(pat, "", s)
    # Remove empty punctuation.
    s = re.sub(r"[，；;]{2,}", "，", s)
    return s.strip("，；;。 ")


def detect_subject(s: str) -> str:
    # Pick the first person word appearing in the main text. This avoids choosing
    # a later secondary target such as “小孩” in “一个...的人，抱着...小孩”.
    hits = []
    for w in PERSON_WORDS:
        idx = s.find(w)
        if idx >= 0:
            hits.append((idx, -len(w), w))
    if not hits:
        return "人"
    _, _, w = sorted(hits)[0]
    return "人" if w == "行人" else w


def remove_secondary_target_clauses(s: str) -> str:
    # Remove clauses that usually describe another person rather than the main target.
    patterns = [
        r"抱[着]?[^，。；;]*(?:小孩|女孩|男孩|儿童)[^，。；;]*",
        r"旁边有[^，。；;]*(?:人|男子|男人|女人|女子|女孩|男孩|小孩|儿童)[^，。；;]*",
        r"后座[^，。；;]*(?:小孩|女孩|男孩|儿童)[^，。；;]*",
        r"载[着]?[^，。；;]*(?:小孩|女孩|男孩|儿童)[^，。；;]*",
    ]
    for pat in patterns:
        s = re.sub(pat, "", s)
    s = re.sub(r"[，；;]{2,}", "，", s)
    return s.strip("，；;。 ")


def uniq_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        x = normalize_query(x)
        if not x:
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def normalize_query(q: str) -> str:
    q = normalize_text(q)
    q = q.replace("一个", "")
    q = q.replace("有阿迪图标的", "阿迪图标")
    q = re.sub(r"，+", "，", q)
    q = q.strip("，；;。 ")
    # Avoid weird short/fragment-only queries.
    if len(q) < 4:
        return ""
    return q


def color_re() -> str:
    long = "|".join(map(re.escape, COLORS))
    short = "|".join(map(re.escape, SHORT_COLOR_MAP.keys()))
    return f"(?:{long}|{short})"


def canonical_color(c: str) -> str:
    return SHORT_COLOR_MAP.get(c, c)


def find_color_object_attrs(s: str) -> dict[str, list[str]]:
    """Extract color-object attributes grouped by semantic role."""
    attrs: dict[str, list[str]] = defaultdict(list)
    cr = color_re()

    # Upper clothing patterns.
    for m in re.finditer(rf"({cr})(?:的)?(?:有[^，。；;]*?图标的|阿迪图标)?(短袖|长袖|上衣|衣服|外套|雨衣)", s):
        color, obj = canonical_color(m.group(1)), m.group(2)
        attrs["upper"].append(f"穿{color}{obj}")
    for m in re.finditer(rf"穿({cr})(?:的)?(?:有[^，。；;]*?图标的|阿迪图标)?(短袖|长袖|上衣|衣服|外套|雨衣)", s):
        color, obj = canonical_color(m.group(1)), m.group(2)
        attrs["upper"].append(f"穿{color}{obj}")

    # Pants.
    for m in re.finditer(rf"({cr})(?:的)?(裤子|短裤)", s):
        color, obj = canonical_color(m.group(1)), m.group(2)
        attrs["pants"].append(f"穿{color}{obj}")

    # Shoes.
    for m in re.finditer(rf"({cr})(?:的)?(鞋子|鞋)", s):
        color, obj = canonical_color(m.group(1)), m.group(2)
        attrs["shoes"].append(f"穿{color}{obj}")

    # Mask / helmet / hat.
    for m in re.finditer(rf"戴({cr})(?:的)?口罩", s):
        attrs["mask"].append(f"戴{canonical_color(m.group(1))}口罩")
    for m in re.finditer(rf"({cr})(?:的)?口罩", s):
        attrs["mask"].append(f"戴{canonical_color(m.group(1))}口罩")
    for m in re.finditer(rf"戴({cr})(?:的)?头盔", s):
        attrs["helmet"].append(f"戴{canonical_color(m.group(1))}头盔")
    for m in re.finditer(rf"({cr})(?:的)?头盔", s):
        attrs["helmet"].append(f"戴{canonical_color(m.group(1))}头盔")
    for m in re.finditer(rf"戴({cr})(?:的)?帽子", s):
        attrs["hat"].append(f"戴{canonical_color(m.group(1))}帽子")
    for m in re.finditer(rf"({cr})(?:的)?帽子", s):
        attrs["hat"].append(f"戴{canonical_color(m.group(1))}帽子")

    # Bags.  Keep 包/袋子 but avoid turning every 包 into 背包 if original says 手提袋/袋子.
    for m in re.finditer(rf"(?:背|背着|背了)?({cr})(?:的)?(书包|背包)", s):
        color, obj = canonical_color(m.group(1)), m.group(2)
        attrs["bag"].append(f"背{color}{obj}")
    for m in re.finditer(rf"({cr})(?:的)?(袋子|手提袋)", s):
        color, obj = canonical_color(m.group(1)), m.group(2)
        attrs["bag"].append(f"拿{color}{obj}")

    # Vehicles / windshield.
    for m in re.finditer(rf"骑(?:着)?({cr})?(?:的)?(电动车|自行车|摩托车)", s):
        color, obj = m.group(1), m.group(2)
        if color:
            attrs["vehicle"].append(f"骑{canonical_color(color)}{obj}")
        else:
            attrs["vehicle"].append(f"骑{obj}")
    for m in re.finditer(rf"({cr})(?:的)?(电动车|自行车|摩托车)", s):
        color, obj = canonical_color(m.group(1)), m.group(2)
        attrs["vehicle"].append(f"骑{color}{obj}")
    for m in re.finditer(rf"({cr})(?:的)?挡风被", s):
        attrs["windshield"].append(f"有{canonical_color(m.group(1))}挡风被")

    # Actions / behavior labels.
    if "抽烟" in s or "吸烟" in s:
        attrs["action"].append("抽烟")
    if "打电话" in s or "接电话" in s or "玩手机" in s:
        attrs["action"].append("打电话")

    return {k: uniq_keep_order(v) for k, v in attrs.items() if v}


def flatten_attrs(attrs: dict[str, list[str]]) -> list[str]:
    out = []
    for g in GROUP_ORDER:
        out.extend(attrs.get(g, []))
    return uniq_keep_order(out)


def combine(attrs: list[str], subject: str, max_combos: int = 8) -> list[str]:
    """Generate short attribute combinations without exploding too much."""
    qs = []
    # Single attribute queries.
    for a in attrs:
        qs.append(f"{a}的{subject}")
    # Useful pairs: upper+pants, upper+shoes, upper+vehicle, pants+vehicle, helmet+vehicle, mask+upper, bag+upper.
    def get(prefix: str) -> list[str]:
        return [a for a in attrs if a.startswith(prefix)]

    uppers = [a for a in attrs if any(x in a for x in ["上衣", "短袖", "长袖", "衣服", "外套", "雨衣"])]
    pants = [a for a in attrs if "裤" in a]
    shoes = [a for a in attrs if "鞋" in a]
    masks = [a for a in attrs if "口罩" in a]
    helmets = [a for a in attrs if "头盔" in a]
    hats = [a for a in attrs if "帽子" in a]
    bags = [a for a in attrs if "书包" in a or "背包" in a or "袋" in a]
    vehicles = [a for a in attrs if "骑" in a and any(v in a for v in ["电动车", "自行车", "摩托车"])]
    windshields = [a for a in attrs if "挡风被" in a]
    actions = [a for a in attrs if a in ["抽烟", "打电话"]]

    pair_groups = [
        (uppers, pants), (uppers, shoes), (pants, shoes), (masks, uppers), (helmets, uppers),
        (hats, uppers), (bags, uppers), (uppers, vehicles), (pants, vehicles), (helmets, vehicles),
        (vehicles, windshields), (actions, uppers), (actions, vehicles),
    ]
    for lefts, rights in pair_groups:
        for a in lefts[:2]:
            for b in rights[:2]:
                if a != b:
                    qs.append(f"{a}{b}的{subject}")

    # A compact full-attribute query: first 3-4 most important attrs.
    if len(attrs) >= 3:
        qs.append("".join(attrs[:4]) + f"的{subject}")

    return uniq_keep_order(qs)[:max_combos]


def expand_one(text: str, max_new_per_item: int = 10) -> list[str]:
    raw = normalize_text(text)
    cleaned = clean_background(raw)
    main_cleaned = remove_secondary_target_clauses(cleaned)
    subject = detect_subject(main_cleaned)
    attrs_by_group = find_color_object_attrs(main_cleaned)
    attrs = flatten_attrs(attrs_by_group)

    queries = []
    # Cleaned full query often keeps the original meaning but drops background clutter.
    if main_cleaned and main_cleaned != raw:
        queries.append(main_cleaned)
    # If original is not too long, keep a normalized version among generated variants too.
    if len(raw) <= 45:
        queries.append(raw)

    queries.extend(combine(attrs, subject, max_combos=max_new_per_item))

    # Extra vehicle-target phrase when text clearly has rider and windshield.
    if attrs_by_group.get("vehicle") and attrs_by_group.get("windshield"):
        for v in attrs_by_group["vehicle"][:1]:
            for w in attrs_by_group["windshield"][:1]:
                queries.append(f"{v}{w}的{subject}")

    # Avoid returning exact original as added item; original will be kept separately.
    queries = [q for q in uniq_keep_order(queries) if q != raw]
    return queries[:max_new_per_item]


def expand_rows(rows: list[dict], max_new_per_item: int, keep_original: bool) -> tuple[list[dict], list[int]]:
    out = []
    seen_pairs = set()
    next_id = 0
    added_counts = []

    def add(text: str, image_ids: list[int], source_text_id=None, expanded=False):
        nonlocal next_id
        text = normalize_query(text)
        if not text:
            return False
        key = (text, tuple(image_ids))
        if key in seen_pairs:
            return False
        seen_pairs.add(key)
        item = {"text_id": next_id, "text": text, "image_ids": image_ids}
        if expanded:
            item["source_text_id"] = source_text_id
        out.append(item)
        next_id += 1
        return True

    for row in rows:
        text = row.get("text", "")
        image_ids = row.get("image_ids", [])
        if keep_original:
            add(text, image_ids, row.get("text_id"), expanded=False)
        before = len(out)
        for q in expand_one(text, max_new_per_item):
            add(q, image_ids, row.get("text_id"), expanded=True)
        added_counts.append(len(out) - before)

    return out, added_counts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def copy_dataset_files(src_dir: Path, out_dir: Path, skip_names: set[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.iterdir()):
        if src.is_file() and src.name not in skip_names:
            shutil.copy2(src, out_dir / src.name)


def print_stats(input_rows: int, out: list[dict], added_counts: list[int]) -> None:
    c = Counter(added_counts)
    print(f"input_rows={input_rows}")
    print(f"output_rows={len(out)}")
    print(f"expanded_rows={sum(1 for x in out if 'source_text_id' in x)}")
    print(f"avg_new_per_input={sum(added_counts)/max(len(added_counts),1):.2f}")
    print(f"new_count_distribution={dict(sorted(c.items()))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Input train_texts.jsonl or a MUGE dataset directory.")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL path or dataset directory.")
    ap.add_argument("--split", default="train", help="Split to expand when --input is a dataset directory. Default: train.")
    ap.add_argument("--max-new-per-item", type=int, default=8)
    ap.add_argument("--keep-original", action="store_true", default=True)
    ap.add_argument("--no-keep-original", dest="keep_original", action="store_false")
    ap.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output path.")
    args = ap.parse_args()

    if args.input.is_dir():
        input_texts = args.input / f"{args.split}_texts.jsonl"
        if not input_texts.is_file():
            raise FileNotFoundError(f"Missing text file: {input_texts}")
        if args.output.exists() and not args.output.is_dir():
            raise NotADirectoryError(f"Dataset output must be a directory: {args.output}")
        if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {args.output}. Use --overwrite to replace files.")

        rows = load_jsonl(input_texts)
        out, added_counts = expand_rows(rows, args.max_new_per_item, args.keep_original)
        copy_dataset_files(args.input, args.output, skip_names={f"{args.split}_texts.jsonl"})
        write_jsonl(args.output / f"{args.split}_texts.jsonl", out)
        print(f"output_dataset={args.output.resolve()}")
        print_stats(len(rows), out, added_counts)
        return

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {args.output}. Use --overwrite to replace it.")
    rows = load_jsonl(args.input)
    out, added_counts = expand_rows(rows, args.max_new_per_item, args.keep_original)
    write_jsonl(args.output, out)

    print_stats(len(rows), out, added_counts)


if __name__ == "__main__":
    main()
