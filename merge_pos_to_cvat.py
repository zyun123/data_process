#!/usr/bin/env python3
"""
Merge train/valid data from pos_datasets into cvat_datasets and write a new dataset directory.

The script appends:
  - pos train_texts.jsonl -> cvat train_texts.jsonl
  - pos train_imgs.tsv    -> cvat train_imgs.tsv
  - pos valid_texts.jsonl -> cvat valid_texts.jsonl
  - pos valid_imgs.tsv    -> cvat valid_imgs.tsv

Positive-sample text IDs are rewritten per split to continue after the max IDs
in the matching CVAT split. Image IDs are rewritten globally to continue after
the max IDs in all CVAT image TSV files.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge pos_datasets train/valid splits into cvat_datasets.")
    parser.add_argument("--cvat-dir", type=Path, default=Path("cvat_datasets"), help="Base CVAT MUGE dataset dir.")
    parser.add_argument("--pos-dir", type=Path, default=Path("pos_datasets"), help="Positive MUGE train dataset dir.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cvat_pos_datasets"),
        help="Output dataset directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")


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


def read_tsv(path: Path) -> List[Tuple[int, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid TSV row in {path}:{line_no}")
            try:
                image_id = int(parts[0])
            except ValueError as exc:
                raise ValueError(f"Invalid image_id in {path}:{line_no}: {parts[0]!r}") from exc
            rows.append((image_id, parts[1]))
    return rows


def write_tsv(path: Path, rows: Iterable[Tuple[int, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for image_id, image_b64 in rows:
            f.write(f"{image_id}\t{image_b64}\n")


def max_text_id(records: Iterable[dict]) -> int:
    ids = [int(record["text_id"]) for record in records]
    return max(ids, default=-1)


def max_image_id(rows: Iterable[Tuple[int, str]]) -> int:
    ids = [image_id for image_id, _ in rows]
    return max(ids, default=0)


def max_dataset_image_id(dataset_dir: Path) -> int:
    max_id = 0
    for path in sorted(dataset_dir.glob("*_imgs.tsv")):
        max_id = max(max_id, max_image_id(read_tsv(path)))
    return max_id


def build_image_id_map(pos_img_rows: List[Tuple[int, str]], start_id: int) -> Dict[int, int]:
    mapping = {}
    next_id = start_id
    for old_id, _ in pos_img_rows:
        if old_id in mapping:
            raise ValueError(f"Duplicate image_id in pos image TSV files: {old_id}")
        mapping[old_id] = next_id
        next_id += 1
    return mapping


def remap_pos_texts(pos_texts: List[dict], image_id_map: Dict[int, int], text_start_id: int, split: str) -> List[dict]:
    remapped = []
    for offset, record in enumerate(pos_texts):
        old_image_ids = record.get("image_ids", [])
        missing = [image_id for image_id in old_image_ids if image_id not in image_id_map]
        if missing:
            raise ValueError(
                f"pos {split} text_id {record.get('text_id')} references image_ids missing from {split}_imgs.tsv: {missing}"
            )

        new_record = dict(record)
        new_record["text_id"] = text_start_id + offset
        new_record["image_ids"] = [image_id_map[image_id] for image_id in old_image_ids]
        remapped.append(new_record)
    return remapped


def validate_split(texts: List[dict], imgs: List[Tuple[int, str]], split: str) -> None:
    image_ids = [image_id for image_id, _ in imgs]
    duplicate_image_ids = sorted({image_id for image_id in image_ids if image_ids.count(image_id) > 1})
    if duplicate_image_ids:
        raise ValueError(f"Duplicate image_id in output {split}_imgs.tsv: {duplicate_image_ids[:10]}")

    image_id_set = set(image_ids)
    missing = []
    for record in texts:
        for image_id in record.get("image_ids", []):
            if int(image_id) not in image_id_set:
                missing.append((record.get("text_id"), image_id))
    if missing:
        raise ValueError(f"Output {split}_texts.jsonl references missing image_ids: {missing[:10]}")


def copy_cvat_other_files(cvat_dir: Path, out_dir: Path) -> None:
    for src in sorted(cvat_dir.iterdir()):
        if not src.is_file():
            continue
        if src.name in {"train_texts.jsonl", "train_imgs.tsv", "valid_texts.jsonl", "valid_imgs.tsv"}:
            continue
        shutil.copy2(src, out_dir / src.name)


def read_split(dataset_dir: Path, split: str, required: bool) -> Tuple[List[dict], List[Tuple[int, str]]]:
    texts_path = dataset_dir / f"{split}_texts.jsonl"
    imgs_path = dataset_dir / f"{split}_imgs.tsv"
    if not texts_path.is_file() or not imgs_path.is_file():
        if required:
            require_file(texts_path)
            require_file(imgs_path)
        return [], []
    return read_jsonl(texts_path), read_tsv(imgs_path)


def remap_split(
    pos_texts: List[dict],
    pos_imgs: List[Tuple[int, str]],
    image_id_map: Dict[int, int],
    text_start_id: int,
    split: str,
) -> Tuple[List[dict], List[Tuple[int, str]]]:
    remapped_imgs = [(image_id_map[old_id], image_b64) for old_id, image_b64 in pos_imgs]
    remapped_texts = remap_pos_texts(pos_texts, image_id_map, text_start_id, split)
    return remapped_texts, remapped_imgs


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

    all_pos_imgs = [*pos_train_imgs, *pos_valid_imgs]
    image_id_map = build_image_id_map(all_pos_imgs, max_dataset_image_id(cvat_dir) + 1)
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
        max_text_id(cvat_valid_texts) + 1,
        "valid",
    )

    output_train_texts = [*cvat_train_texts, *remapped_pos_train_texts]
    output_train_imgs = [*cvat_train_imgs, *remapped_pos_train_imgs]
    output_valid_texts = [*cvat_valid_texts, *remapped_pos_valid_texts]
    output_valid_imgs = [*cvat_valid_imgs, *remapped_pos_valid_imgs]

    validate_split(output_train_texts, output_train_imgs, "train")
    validate_split(output_valid_texts, output_valid_imgs, "valid")

    copy_cvat_other_files(cvat_dir, out_dir)
    write_jsonl(out_dir / "train_texts.jsonl", output_train_texts)
    write_tsv(out_dir / "train_imgs.tsv", output_train_imgs)
    write_jsonl(out_dir / "valid_texts.jsonl", output_valid_texts)
    write_tsv(out_dir / "valid_imgs.tsv", output_valid_imgs)

    print(f"wrote output dir: {out_dir}")
    print(
        f"train texts: {len(cvat_train_texts)} + {len(remapped_pos_train_texts)} = {len(output_train_texts)}"
    )
    print(f"train images: {len(cvat_train_imgs)} + {len(remapped_pos_train_imgs)} = {len(output_train_imgs)}")
    print(
        f"valid texts: {len(cvat_valid_texts)} + {len(remapped_pos_valid_texts)} = {len(output_valid_texts)}"
    )
    print(f"valid images: {len(cvat_valid_imgs)} + {len(remapped_pos_valid_imgs)} = {len(output_valid_imgs)}")
    if remapped_pos_train_texts:
        print(f"added train text_id range: {remapped_pos_train_texts[0]['text_id']}..{remapped_pos_train_texts[-1]['text_id']}")
    if remapped_pos_valid_texts:
        print(f"added valid text_id range: {remapped_pos_valid_texts[0]['text_id']}..{remapped_pos_valid_texts[-1]['text_id']}")
    if image_id_map:
        print(f"added image_id range: {min(image_id_map.values())}..{max(image_id_map.values())}")


if __name__ == "__main__":
    main()
