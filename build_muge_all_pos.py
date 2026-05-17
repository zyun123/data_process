#!/usr/bin/env python3
import argparse
import base64
import json
import random
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory: Path):
    if not directory.is_dir():
        return []
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=lambda p: p.name,
    )


def read_query(sample_dir: Path):
    query_path = sample_dir / "query.txt"
    if not query_path.is_file():
        return ""
    return query_path.read_text(encoding="utf-8").strip()


def iter_sample_dirs(root: Path):
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


def encode_image(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def split_sample_dirs(sample_dirs, valid_ratio: float, seed: int):
    if not (0 <= valid_ratio < 1):
        raise ValueError("valid-ratio must be in [0, 1).")
    if len(sample_dirs) < 2 or valid_ratio == 0:
        return sample_dirs, []

    idxs = list(range(len(sample_dirs)))
    random.Random(seed).shuffle(idxs)
    valid_n = int(round(len(sample_dirs) * valid_ratio))
    if valid_n == 0:
        valid_n = 1
    if valid_n >= len(sample_dirs):
        valid_n = len(sample_dirs) - 1

    valid_idxs = set(idxs[:valid_n])
    train_dirs = [sample_dir for i, sample_dir in enumerate(sample_dirs) if i not in valid_idxs]
    valid_dirs = [sample_dir for i, sample_dir in enumerate(sample_dirs) if i in valid_idxs]
    return train_dirs, valid_dirs


def build_split(sample_dirs, text_start_id: int, image_start_id: int):
    text_records = []
    image_records = []
    skipped = []

    text_id = text_start_id
    image_id = image_start_id

    for sample_dir in sample_dirs:
        text = read_query(sample_dir)
        pos_images = list_images(sample_dir / "pos_images")

        if not text:
            skipped.append((sample_dir.name, "empty or missing query.txt"))
            continue
        if not pos_images:
            skipped.append((sample_dir.name, "no positive images"))
            continue

        image_ids = []
        for image_path in pos_images:
            image_ids.append(image_id)
            image_records.append((image_id, image_path))
            image_id += 1

        text_records.append(
            {
                "text_id": text_id,
                "text": text,
                "image_ids": image_ids,
            }
        )
        text_id += 1

    return text_records, image_records, skipped


def build_dataset(root: Path, text_start_id: int, image_start_id: int, valid_ratio: float, seed: int):
    sample_dirs = iter_sample_dirs(root)
    train_dirs, valid_dirs = split_sample_dirs(sample_dirs, valid_ratio, seed)

    train_texts, train_images, train_skipped = build_split(train_dirs, text_start_id, image_start_id)
    next_text_id = text_start_id + len(train_texts)
    next_image_id = image_start_id + len(train_images)
    valid_texts, valid_images, valid_skipped = build_split(valid_dirs, next_text_id, next_image_id)

    skipped = [("train", *item) for item in train_skipped]
    skipped.extend(("valid", *item) for item in valid_skipped)
    return train_texts, train_images, valid_texts, valid_images, skipped


def write_texts(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_images(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for image_id, image_path in records:
            f.write(f"{image_id}\t{encode_image(image_path)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build a MUGE-format pos dataset using all pos_images for each query."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Directory containing 000001-style folders.")
    parser.add_argument("--out-dir", type=Path, default=Path("muge_all_pos"), help="Output dataset directory.")
    parser.add_argument("--text-start-id", type=int, default=0, help="First text_id.")
    parser.add_argument("--image-start-id", type=int, default=0, help="First image_id.")
    parser.add_argument("--valid-ratio", type=float, default=0.0, help="Directory/query-level validation ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/valid split.")
    args = parser.parse_args()

    train_texts, train_images, valid_texts, valid_images, skipped = build_dataset(
        args.root,
        args.text_start_id,
        args.image_start_id,
        args.valid_ratio,
        args.seed,
    )
    if not train_texts:
        raise SystemExit("No valid positive samples found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_texts(args.out_dir / "train_texts.jsonl", train_texts)
    write_images(args.out_dir / "train_imgs.tsv", train_images)
    write_texts(args.out_dir / "valid_texts.jsonl", valid_texts)
    write_images(args.out_dir / "valid_imgs.tsv", valid_images)

    print(f"wrote train texts: {args.out_dir / 'train_texts.jsonl'} ({len(train_texts)} lines)")
    print(f"wrote train images: {args.out_dir / 'train_imgs.tsv'} ({len(train_images)} lines)")
    print(f"wrote valid texts: {args.out_dir / 'valid_texts.jsonl'} ({len(valid_texts)} lines)")
    print(f"wrote valid images: {args.out_dir / 'valid_imgs.tsv'} ({len(valid_images)} lines)")
    print(f"train text_id range: {train_texts[0]['text_id']}..{train_texts[-1]['text_id']}")
    print(f"train image_id range: {train_images[0][0]}..{train_images[-1][0]}")
    if valid_texts:
        print(f"valid text_id range: {valid_texts[0]['text_id']}..{valid_texts[-1]['text_id']}")
    if valid_images:
        print(f"valid image_id range: {valid_images[0][0]}..{valid_images[-1][0]}")
    print(f"skipped dirs: {len(skipped)}")
    for split, sample_id, reason in skipped:
        print(f"  {split}/{sample_id}: {reason}")


if __name__ == "__main__":
    main()
