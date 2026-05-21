#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_tsv(path: Path) -> dict[int, str]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid TSV row in {path}:{line_no}")
            image_id = int(parts[0])
            if image_id in rows:
                raise ValueError(f"Duplicate image_id in {path}: {image_id}")
            rows[image_id] = parts[1]
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: dict[int, str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for image_id in sorted(rows):
            f.write(f"{image_id}\t{rows[image_id]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a candidate-only split for hard-negative retrieval eval.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datapath/datasets/MUGE"))
    parser.add_argument("--source-split", default="valid")
    parser.add_argument("--out-split", default="valid_hard_eval")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    source = args.source_split
    out = args.out_split

    positives = read_tsv(dataset_dir / f"{source}_imgs.tsv")
    negatives = read_tsv(dataset_dir / f"{source}_hard_neg_imgs.tsv")
    hard_rows = read_jsonl(dataset_dir / f"{source}_hard_negatives.jsonl")

    candidate_images: dict[int, str] = {}
    eval_texts = []
    eval_meta = []
    for row in hard_rows:
        text_id = int(row["text_id"])
        pos_ids = [int(image_id) for image_id in row.get("positive_image_ids", [])]
        neg_ids = [int(image_id) for image_id in row.get("negative_image_ids", [])]
        if not pos_ids or not neg_ids:
            continue

        missing_pos = [image_id for image_id in pos_ids if image_id not in positives]
        missing_neg = [image_id for image_id in neg_ids if image_id not in negatives]
        if missing_pos:
            raise ValueError(f"text_id={text_id} missing positive image ids from {source}_imgs.tsv: {missing_pos[:10]}")
        if missing_neg:
            raise ValueError(f"text_id={text_id} missing negative image ids from {source}_hard_neg_imgs.tsv: {missing_neg[:10]}")

        for image_id in pos_ids:
            candidate_images[image_id] = positives[image_id]
        for image_id in neg_ids:
            candidate_images[image_id] = negatives[image_id]

        eval_texts.append(
            {
                "text_id": text_id,
                "text": row["text"],
                "image_ids": pos_ids,
            }
        )
        eval_meta.append(
            {
                "text_id": text_id,
                "text": row["text"],
                "positive_image_ids": pos_ids,
                "negative_image_ids": neg_ids,
                "candidate_image_ids": pos_ids + neg_ids,
            }
        )

    write_tsv(dataset_dir / f"{out}_imgs.tsv", candidate_images)
    write_jsonl(dataset_dir / f"{out}_texts.jsonl", eval_texts)
    write_jsonl(dataset_dir / f"{out}_meta.jsonl", eval_meta)

    print(f"source hard-negative rows: {len(hard_rows)}")
    print(f"eval texts: {len(eval_texts)}")
    print(f"candidate images: {len(candidate_images)}")
    print(f"wrote: {dataset_dir / f'{out}_texts.jsonl'}")
    print(f"wrote: {dataset_dir / f'{out}_imgs.tsv'}")
    print(f"wrote: {dataset_dir / f'{out}_meta.jsonl'}")


if __name__ == "__main__":
    main()
