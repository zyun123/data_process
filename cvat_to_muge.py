#!/usr/bin/env python3
"""
Convert CVAT COCO exports to Chinese-CLIP MUGE-style dataset files.

Output files:
  - train_imgs.tsv
  - train_texts.jsonl
  - valid_imgs.tsv
  - valid_texts.jsonl
"""

import argparse
import base64
import json
import random
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CVAT COCO to MUGE format.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more CVAT COCO export directories (each contains images/ and annotations/instances_default.json).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output dataset directory, e.g. datasets/MUGE_custom",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation split ratio (0~1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument(
        "--caption-joiner",
        default="；",
        help="String used to join multiple boxes' texts in one image.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep images without caption text (fallback to category names).",
    )
    return parser.parse_args()


def encode_image_to_b64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        img = Image.open(f)
        img_buffer = BytesIO()
        fmt = img.format if img.format else "JPEG"
        img.save(img_buffer, format=fmt)
    return base64.b64encode(img_buffer.getvalue()).decode("utf-8")


def normalize_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def load_coco(input_dir: Path) -> List[Dict]:
    ann_path = input_dir / "annotations" / "instances_default.json"
    img_dir = input_dir / "images"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {img_dir}")

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {c["id"]: c.get("name", "") for c in coco.get("categories", [])}
    images = {im["id"]: im for im in coco.get("images", [])}

    texts_by_image: Dict[int, List[str]] = {image_id: [] for image_id in images}

    for ann in coco.get("annotations", []):
        image_id = ann.get("image_id")
        attrs = ann.get("attributes", {}) or {}
        txt = normalize_text(attrs.get("txt", ""))
        if not txt:
            # txt = normalize_text(categories.get(ann.get("category_id"), ""))
            continue
        if txt:
            texts_by_image.setdefault(image_id, []).append(txt)

    samples: List[Dict] = []
    prefix = input_dir.name.replace("\t", " ")

    for image_id, im in images.items():
        file_name = im.get("file_name")
        if not file_name:
            continue
        image_path = img_dir / file_name
        if not image_path.exists():
            continue

        raw_texts = texts_by_image.get(image_id, [])
        dedup_texts: List[str] = []
        seen = set()
        for t in raw_texts:
            if t not in seen:
                dedup_texts.append(t)
                seen.add(t)

        samples.append(
            {
                "uid": f"{prefix}__{image_id}",
                "image_path": image_path,
                "texts": dedup_texts,
            }
        )

    return samples


def build_samples(input_dirs: List[Path], joiner: str, keep_empty: bool) -> List[Dict]:
    merged: List[Dict] = []
    for d in input_dirs:
        merged.extend(load_coco(d))

    final = []
    for item in merged:
        caption = joiner.join(item["texts"]).strip()
        if not caption and not keep_empty:
            continue
        final.append({"uid": item["uid"], "image_path": item["image_path"], "caption": caption})

    for idx, item in enumerate(final, start=1):
        item["image_id"] = idx
    return final


def split_train_valid(samples: List[Dict], valid_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if not (0 <= valid_ratio < 1):
        raise ValueError("valid_ratio must be in [0, 1).")
    if len(samples) < 2:
        return samples, []

    idxs = list(range(len(samples)))
    random.Random(seed).shuffle(idxs)

    valid_n = int(round(len(samples) * valid_ratio))
    if valid_ratio > 0 and valid_n == 0:
        valid_n = 1
    if valid_n >= len(samples):
        valid_n = len(samples) - 1

    valid_set = set(idxs[:valid_n])
    train, valid = [], []
    for i, sample in enumerate(samples):
        (valid if i in valid_set else train).append(sample)
    return train, valid


def write_split(split_name: str, rows: List[Dict], output_dir: Path) -> None:
    imgs_path = output_dir / f"{split_name}_imgs.tsv"
    texts_path = output_dir / f"{split_name}_texts.jsonl"

    with imgs_path.open("w", encoding="utf-8") as f_img, texts_path.open("w", encoding="utf-8") as f_txt:
        for text_id, row in enumerate(rows):
            image_id = row["image_id"]
            b64 = encode_image_to_b64(row["image_path"])
            f_img.write(f"{image_id}\t{b64}\n")

            item = {
                "text_id": text_id,
                "text": row["caption"],
                "image_ids": [image_id],
            }
            f_txt.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    # input_dirs = [Path(p).expanduser().resolve() for p in args.inputs]
    input_dirs = [
        Path("/home/zy/Downloads/data_process/cvat_data/task_2026-04-29").expanduser().resolve(),
        Path("/home/zy/Downloads/data_process/cvat_data/task_2026-04-29-2").expanduser().resolve(),
        Path("/home/zy/Downloads/data_process/cvat_data/task_2026-05-07").expanduser().resolve(),
        Path("/home/zy/Downloads/data_process/cvat_data/task_2026-05-11").expanduser().resolve()
    ]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(input_dirs, args.caption_joiner, args.keep_empty)
    if not samples:
        raise RuntimeError("No valid samples found. Check input paths or annotations.")

    train_rows, valid_rows = split_train_valid(samples, args.valid_ratio, args.seed)
    write_split("train", train_rows, output_dir)
    write_split("valid", valid_rows, output_dir)

    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_rows)}")
    print(f"Valid samples: {len(valid_rows)}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
