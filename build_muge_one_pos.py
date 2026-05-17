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


def find_pos_images(sample_dir: Path):
    for name in ("pos_images", "pos"):
        images = list_images(sample_dir / name)
        if images:
            return images
    return []


def read_query(sample_dir: Path):
    query_path = sample_dir / "query.txt"
    if not query_path.is_file():
        return ""
    return query_path.read_text(encoding="utf-8").strip()


def max_text_id(path: Path):
    if not path.is_file():
        return -1

    max_id = -1
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            max_id = max(max_id, int(item["text_id"]))
    return max_id


def max_image_id(path: Path):
    if not path.is_file():
        return 0

    max_id = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id = line.split("\t", 1)[0]
            max_id = max(max_id, int(image_id))
    return max_id


def infer_start_ids(ref_dir: Path, split: str):
    text_start = max_text_id(ref_dir / f"{split}_texts.jsonl") + 1
    image_start = max(
        max_image_id(ref_dir / "train_imgs.tsv"),
        max_image_id(ref_dir / "valid_imgs.tsv"),
    ) + 1
    return text_start, image_start


def iter_sample_dirs(root: Path, start: int, end: int):
    for i in range(start, end + 1):
        sample_dir = root / f"{i:06d}"
        if sample_dir.is_dir():
            yield sample_dir


def encode_image(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def build_records(root: Path, start: int, end: int, text_start: int, image_start: int):
    records = []
    images = []
    skipped = []

    text_id = text_start
    image_id = image_start
    for sample_dir in iter_sample_dirs(root, start, end):
        query = read_query(sample_dir)
        pos_images = find_pos_images(sample_dir)

        if not query:
            skipped.append((sample_dir.name, "missing query.txt or empty query"))
            continue
        if not pos_images:
            skipped.append((sample_dir.name, "no pos image"))
            continue

        pos_image = pos_images[0]
        records.append(
            {
                "text_id": text_id,
                "text": query,
                "image_ids": [image_id],
            }
        )
        images.append((image_id, pos_image))
        text_id += 1
        image_id += 1

    return records, images, skipped


def write_texts(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_images(path: Path, images):
    with path.open("w", encoding="utf-8") as f:
        for image_id, image_path in images:
            f.write(f"{image_id}\t{encode_image(image_path)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build MUGE-format texts jsonl and imgs tsv from query.txt plus one pos image."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Directory containing 000001-style folders.")
    parser.add_argument("--out-dir", type=Path, default=Path("muge_one_pos"), help="Output directory.")
    parser.add_argument("--start", type=int, default=1, help="First numeric sample id, inclusive.")
    parser.add_argument("--end", type=int, default=137, help="Last numeric sample id, inclusive.")
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=Path("/home/zy/vision/Chinese-CLIP/datasets/muge_from_cvat"),
        help="Existing MUGE directory used to infer append-safe ids.",
    )
    parser.add_argument("--split", choices=("train", "valid"), default="train", help="Split whose text ids will be continued.")
    parser.add_argument("--text-start-id", type=int, default=None, help="Override first text_id.")
    parser.add_argument("--image-start-id", type=int, default=None, help="Override first image_id.")
    parser.add_argument("--texts-name", default=None, help="Output jsonl filename. Default: new_<split>_texts.jsonl")
    parser.add_argument("--imgs-name", default=None, help="Output tsv filename. Default: new_<split>_imgs.tsv")
    args = parser.parse_args()

    inferred_text_start, inferred_image_start = infer_start_ids(args.ref_dir, args.split)
    text_start = args.text_start_id if args.text_start_id is not None else inferred_text_start
    image_start = args.image_start_id if args.image_start_id is not None else inferred_image_start

    records, images, skipped = build_records(args.root, args.start, args.end, text_start, image_start)
    if not records:
        raise SystemExit("No valid samples found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    texts_name = args.texts_name or f"new_{args.split}_texts.jsonl"
    imgs_name = args.imgs_name or f"new_{args.split}_imgs.tsv"
    texts_path = args.out_dir / texts_name
    imgs_path = args.out_dir / imgs_name

    write_texts(texts_path, records)
    write_images(imgs_path, images)

    print(f"wrote texts: {texts_path} ({len(records)} lines)")
    print(f"wrote images: {imgs_path} ({len(images)} lines)")
    print(f"text_id range: {records[0]['text_id']}..{records[-1]['text_id']}")
    print(f"image_id range: {images[0][0]}..{images[-1][0]}")
    if skipped:
        print(f"skipped: {len(skipped)}")
        for sample_id, reason in skipped:
            print(f"  {sample_id}: {reason}")


if __name__ == "__main__":
    main()
