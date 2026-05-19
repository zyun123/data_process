#!/usr/bin/env python3
import argparse
import base64
import hashlib
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


def find_neg_images(sample_dir: Path):
    for name in ("neg_images", "neg"):
        images = list_images(sample_dir / name)
        if images:
            return images
    return []


def iter_sample_dirs(root: Path):
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


def encode_image(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def image_content_hash(path: Path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def image_hashes_for_sample(sample_dir: Path):
    paths = [*list_images(sample_dir / "pos_images"), *find_neg_images(sample_dir)]
    return {image_content_hash(path) for path in paths}


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def group_sample_dirs_by_image_content(sample_dirs):
    uf = UnionFind(len(sample_dirs))
    owner_by_hash = {}
    for idx, sample_dir in enumerate(sample_dirs):
        for image_hash in image_hashes_for_sample(sample_dir):
            if image_hash in owner_by_hash:
                uf.union(owner_by_hash[image_hash], idx)
            else:
                owner_by_hash[image_hash] = idx

    groups_by_root = {}
    for idx, sample_dir in enumerate(sample_dirs):
        root = uf.find(idx)
        groups_by_root.setdefault(root, []).append(sample_dir)
    return list(groups_by_root.values())


def split_sample_dirs(sample_dirs, valid_ratio: float, seed: int):
    if not (0 <= valid_ratio < 1):
        raise ValueError("valid-ratio must be in [0, 1).")
    if len(sample_dirs) < 2 or valid_ratio == 0:
        return sample_dirs, []

    groups = group_sample_dirs_by_image_content(sample_dirs)
    if len(groups) < 2:
        return sample_dirs, []

    idxs = list(range(len(groups)))
    random.Random(seed).shuffle(idxs)
    valid_target = max(1, int(round(len(sample_dirs) * valid_ratio)))
    valid_n = int(round(len(groups) * valid_ratio))
    if valid_n == 0:
        valid_n = 1
    if valid_n >= len(groups):
        valid_n = len(groups) - 1

    valid_group_idxs = set()
    valid_count = 0
    for group_idx in idxs:
        if len(valid_group_idxs) >= valid_n and valid_count >= valid_target:
            break
        if len(valid_group_idxs) >= len(groups) - 1:
            break
        valid_group_idxs.add(group_idx)
        valid_count += len(groups[group_idx])

    train_dirs = []
    valid_dirs = []
    for group_idx, group in enumerate(groups):
        (valid_dirs if group_idx in valid_group_idxs else train_dirs).extend(group)
    return train_dirs, valid_dirs


def build_split(sample_dirs, text_start_id: int, image_start_id: int, hard_neg_image_start_id: int):
    text_records = []
    image_records = []
    hard_negative_records = []
    hard_negative_image_records = []
    skipped = []

    text_id = text_start_id
    image_id = image_start_id
    hard_neg_image_id = hard_neg_image_start_id

    for sample_dir in sample_dirs:
        text = read_query(sample_dir)
        pos_images = list_images(sample_dir / "pos_images")
        neg_images = find_neg_images(sample_dir)

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

        negative_image_ids = []
        for image_path in neg_images:
            negative_image_ids.append(hard_neg_image_id)
            hard_negative_image_records.append((hard_neg_image_id, image_path))
            hard_neg_image_id += 1

        text_records.append(
            {
                "text_id": text_id,
                "text": text,
                "image_ids": image_ids,
            }
        )
        if negative_image_ids:
            hard_negative_records.append(
                {
                    "text_id": text_id,
                    "text": text,
                    "positive_image_ids": image_ids,
                    "negative_image_ids": negative_image_ids,
                }
            )
        text_id += 1

    return text_records, image_records, hard_negative_records, hard_negative_image_records, skipped


def build_dataset(root: Path, text_start_id: int, image_start_id: int, valid_ratio: float, seed: int):
    sample_dirs = iter_sample_dirs(root)
    train_dirs, valid_dirs = split_sample_dirs(sample_dirs, valid_ratio, seed)

    train_texts, train_images, train_hard_negs, train_hard_neg_images, train_skipped = build_split(
        train_dirs,
        text_start_id,
        image_start_id,
        0,
    )
    next_text_id = text_start_id + len(train_texts)
    next_image_id = image_start_id + len(train_images)
    next_hard_neg_image_id = len(train_hard_neg_images)
    valid_texts, valid_images, valid_hard_negs, valid_hard_neg_images, valid_skipped = build_split(
        valid_dirs,
        next_text_id,
        next_image_id,
        next_hard_neg_image_id,
    )

    skipped = [("train", *item) for item in train_skipped]
    skipped.extend(("valid", *item) for item in valid_skipped)
    return (
        train_texts,
        train_images,
        train_hard_negs,
        train_hard_neg_images,
        valid_texts,
        valid_images,
        valid_hard_negs,
        valid_hard_neg_images,
        skipped,
    )


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

    (
        train_texts,
        train_images,
        train_hard_negs,
        train_hard_neg_images,
        valid_texts,
        valid_images,
        valid_hard_negs,
        valid_hard_neg_images,
        skipped,
    ) = build_dataset(
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
    write_texts(args.out_dir / "train_hard_negatives.jsonl", train_hard_negs)
    write_images(args.out_dir / "train_hard_neg_imgs.tsv", train_hard_neg_images)
    write_texts(args.out_dir / "valid_texts.jsonl", valid_texts)
    write_images(args.out_dir / "valid_imgs.tsv", valid_images)
    write_texts(args.out_dir / "valid_hard_negatives.jsonl", valid_hard_negs)
    write_images(args.out_dir / "valid_hard_neg_imgs.tsv", valid_hard_neg_images)

    print(f"wrote train texts: {args.out_dir / 'train_texts.jsonl'} ({len(train_texts)} lines)")
    print(f"wrote train images: {args.out_dir / 'train_imgs.tsv'} ({len(train_images)} lines)")
    print(f"wrote train hard negatives: {len(train_hard_negs)} texts, {len(train_hard_neg_images)} images")
    print(f"wrote valid texts: {args.out_dir / 'valid_texts.jsonl'} ({len(valid_texts)} lines)")
    print(f"wrote valid images: {args.out_dir / 'valid_imgs.tsv'} ({len(valid_images)} lines)")
    print(f"wrote valid hard negatives: {len(valid_hard_negs)} texts, {len(valid_hard_neg_images)} images")
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
