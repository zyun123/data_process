#!/usr/bin/env python3
import argparse
import base64
import json
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


def build_dataset(root: Path, text_start_id: int, image_start_id: int):
    text_records = []
    image_records = []
    skipped = []

    text_id = text_start_id
    image_id = image_start_id

    for sample_dir in iter_sample_dirs(root):
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
        description="Build a train-only MUGE-format dataset using all pos_images for each query."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Directory containing 000001-style folders.")
    parser.add_argument("--out-dir", type=Path, default=Path("muge_all_pos"), help="Output dataset directory.")
    parser.add_argument("--text-start-id", type=int, default=0, help="First text_id.")
    parser.add_argument("--image-start-id", type=int, default=0, help="First image_id.")
    args = parser.parse_args()

    text_records, image_records, skipped = build_dataset(args.root, args.text_start_id, args.image_start_id)
    if not text_records:
        raise SystemExit("No valid positive samples found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    texts_path = args.out_dir / "train_texts.jsonl"
    imgs_path = args.out_dir / "train_imgs.tsv"

    write_texts(texts_path, text_records)
    write_images(imgs_path, image_records)

    print(f"wrote texts: {texts_path} ({len(text_records)} lines)")
    print(f"wrote images: {imgs_path} ({len(image_records)} lines)")
    print(f"text_id range: {text_records[0]['text_id']}..{text_records[-1]['text_id']}")
    print(f"image_id range: {image_records[0][0]}..{image_records[-1][0]}")
    print(f"skipped dirs: {len(skipped)}")
    for sample_id, reason in skipped:
        print(f"  {sample_id}: {reason}")


if __name__ == "__main__":
    main()
