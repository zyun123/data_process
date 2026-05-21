#!/usr/bin/env python3
"""
Compose the final police retrieval MUGE dataset.

Output layout:
  train_*              CVAT train + stratified pos-query train
  valid_*              stratified pos-query holdout, used as the main validation split
  valid_cvat_long_*    CVAT valid split, kept as an auxiliary long-description benchmark

If the pos holdout contains hard negatives, valid_hard_negatives.jsonl and
valid_hard_neg_imgs.tsv are also written, so it can be used for hard-negative
diagnostics without leaking the held-out query directories into train.
"""

import argparse
from pathlib import Path

from merge_pos_to_cvat import (
    build_image_id_map,
    build_text_id_map,
    copy_cvat_other_files,
    max_dataset_image_id,
    max_image_id,
    max_text_id,
    read_hard_negative_split,
    read_split,
    remap_hard_negatives,
    remap_split,
    require_file,
    validate_split,
    write_jsonl,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose train + stratified validation splits.")
    parser.add_argument("--cvat-dir", type=Path, required=True, help="Filtered CVAT MUGE dataset dir.")
    parser.add_argument("--pos-dir", type=Path, required=True, help="Stratified pos MUGE dataset dir.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output dataset dir.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cvat_dir = args.cvat_dir.expanduser().resolve()
    pos_dir = args.pos_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    for split in ["train", "valid"]:
        for path in [cvat_dir / f"{split}_texts.jsonl", cvat_dir / f"{split}_imgs.tsv"]:
            require_file(path)
    for path in [pos_dir / "train_texts.jsonl", pos_dir / "train_imgs.tsv"]:
        require_file(path)

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite to replace files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    cvat_train_texts, cvat_train_imgs = read_split(cvat_dir, "train", required=True)
    cvat_valid_texts, cvat_valid_imgs = read_split(cvat_dir, "valid", required=True)
    pos_train_texts, pos_train_imgs = read_split(pos_dir, "train", required=True)
    pos_valid_texts, pos_valid_imgs = read_split(pos_dir, "valid", required=False)
    pos_train_hard_negs, pos_train_hard_neg_imgs = read_hard_negative_split(pos_dir, "train")
    pos_valid_hard_negs, pos_valid_hard_neg_imgs = read_hard_negative_split(pos_dir, "valid")

    all_pos_imgs = [*pos_train_imgs, *pos_valid_imgs]
    image_id_map = build_image_id_map(all_pos_imgs, max_dataset_image_id(cvat_dir) + 1)
    all_hard_neg_imgs = [*pos_train_hard_neg_imgs, *pos_valid_hard_neg_imgs]
    next_hard_neg_image_id = max(
        max(image_id_map.values(), default=max_dataset_image_id(cvat_dir)),
        max_image_id(all_hard_neg_imgs),
    ) + 1
    hard_neg_image_id_map = build_image_id_map(all_hard_neg_imgs, next_hard_neg_image_id)

    train_text_id_map = build_text_id_map(pos_train_texts, max_text_id(cvat_train_texts) + 1)
    valid_text_id_map = build_text_id_map(pos_valid_texts, 0)

    remapped_pos_train_texts, remapped_pos_train_imgs = remap_split(
        pos_train_texts,
        pos_train_imgs,
        image_id_map,
        max_text_id(cvat_train_texts) + 1,
        "train",
    )
    remapped_pos_valid_texts, remapped_pos_valid_imgs = remap_split(
        pos_valid_texts,
        pos_valid_imgs,
        image_id_map,
        0,
        "valid",
    )
    remapped_train_hard_negs = remap_hard_negatives(
        pos_train_hard_negs,
        train_text_id_map,
        image_id_map,
        hard_neg_image_id_map,
        "train",
    )
    remapped_valid_hard_negs = remap_hard_negatives(
        pos_valid_hard_negs,
        valid_text_id_map,
        image_id_map,
        hard_neg_image_id_map,
        "valid",
    )
    remapped_train_hard_neg_imgs = [
        (hard_neg_image_id_map[old_id], image_b64) for old_id, image_b64 in pos_train_hard_neg_imgs
    ]
    remapped_valid_hard_neg_imgs = [
        (hard_neg_image_id_map[old_id], image_b64) for old_id, image_b64 in pos_valid_hard_neg_imgs
    ]

    output_train_texts = [*cvat_train_texts, *remapped_pos_train_texts]
    output_train_imgs = [*cvat_train_imgs, *remapped_pos_train_imgs]
    output_valid_texts = remapped_pos_valid_texts
    output_valid_imgs = remapped_pos_valid_imgs

    validate_split(output_train_texts, output_train_imgs, "train")
    validate_split(output_valid_texts, output_valid_imgs, "valid")
    validate_split(cvat_valid_texts, cvat_valid_imgs, "valid_cvat_long")

    copy_cvat_other_files(cvat_dir, out_dir)
    write_jsonl(out_dir / "train_texts.jsonl", output_train_texts)
    write_tsv(out_dir / "train_imgs.tsv", output_train_imgs)
    write_jsonl(out_dir / "train_hard_negatives.jsonl", remapped_train_hard_negs)
    write_tsv(out_dir / "train_hard_neg_imgs.tsv", remapped_train_hard_neg_imgs)

    write_jsonl(out_dir / "valid_texts.jsonl", output_valid_texts)
    write_tsv(out_dir / "valid_imgs.tsv", output_valid_imgs)
    write_jsonl(out_dir / "valid_hard_negatives.jsonl", remapped_valid_hard_negs)
    write_tsv(out_dir / "valid_hard_neg_imgs.tsv", remapped_valid_hard_neg_imgs)

    write_jsonl(out_dir / "valid_cvat_long_texts.jsonl", cvat_valid_texts)
    write_tsv(out_dir / "valid_cvat_long_imgs.tsv", cvat_valid_imgs)

    print(f"wrote output dir: {out_dir}")
    print(f"train texts: {len(cvat_train_texts)} + {len(remapped_pos_train_texts)} = {len(output_train_texts)}")
    print(f"train images: {len(cvat_train_imgs)} + {len(remapped_pos_train_imgs)} = {len(output_train_imgs)}")
    print(f"train hard negatives: {len(remapped_train_hard_negs)} texts/{len(remapped_train_hard_neg_imgs)} images")
    print(f"valid query texts: {len(output_valid_texts)}")
    print(f"valid query images: {len(output_valid_imgs)}")
    print(f"valid hard negatives: {len(remapped_valid_hard_negs)} texts/{len(remapped_valid_hard_neg_imgs)} images")
    print(f"valid_cvat_long texts: {len(cvat_valid_texts)}")
    print(f"valid_cvat_long images: {len(cvat_valid_imgs)}")


if __name__ == "__main__":
    main()
