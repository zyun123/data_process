#!/usr/bin/env python3
"""
Merge train data from pos_datasets into cvat_datasets and write a new dataset directory.

The script appends:
  - pos train_texts.jsonl -> cvat train_texts.jsonl
  - pos train_imgs.tsv    -> cvat train_imgs.tsv

Positive-sample IDs are rewritten to continue after the max IDs in the CVAT train set,
and valid_* files from the CVAT dataset are copied unchanged.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge pos_datasets train split into cvat_datasets.")
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


def build_image_id_map(pos_img_rows: List[Tuple[int, str]], start_id: int) -> Dict[int, int]:
    mapping = {}
    next_id = start_id
    for old_id, _ in pos_img_rows:
        if old_id in mapping:
            raise ValueError(f"Duplicate image_id in pos train_imgs.tsv: {old_id}")
        mapping[old_id] = next_id
        next_id += 1
    return mapping


def remap_pos_texts(pos_texts: List[dict], image_id_map: Dict[int, int], text_start_id: int) -> List[dict]:
    remapped = []
    for offset, record in enumerate(pos_texts):
        old_image_ids = record.get("image_ids", [])
        missing = [image_id for image_id in old_image_ids if image_id not in image_id_map]
        if missing:
            raise ValueError(
                f"pos text_id {record.get('text_id')} references image_ids missing from train_imgs.tsv: {missing}"
            )

        new_record = dict(record)
        new_record["text_id"] = text_start_id + offset
        new_record["image_ids"] = [image_id_map[image_id] for image_id in old_image_ids]
        remapped.append(new_record)
    return remapped


def copy_cvat_non_train_files(cvat_dir: Path, out_dir: Path) -> None:
    for src in sorted(cvat_dir.iterdir()):
        if not src.is_file():
            continue
        if src.name in {"train_texts.jsonl", "train_imgs.tsv"}:
            continue
        shutil.copy2(src, out_dir / src.name)


def main() -> None:
    args = parse_args()
    cvat_dir = args.cvat_dir.expanduser().resolve()
    pos_dir = args.pos_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    cvat_texts_path = cvat_dir / "train_texts.jsonl"
    cvat_imgs_path = cvat_dir / "train_imgs.tsv"
    pos_texts_path = pos_dir / "train_texts.jsonl"
    pos_imgs_path = pos_dir / "train_imgs.tsv"
    for path in [cvat_texts_path, cvat_imgs_path, pos_texts_path, pos_imgs_path]:
        require_file(path)

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite to replace files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    cvat_texts = read_jsonl(cvat_texts_path)
    cvat_imgs = read_tsv(cvat_imgs_path)
    pos_texts = read_jsonl(pos_texts_path)
    pos_imgs = read_tsv(pos_imgs_path)

    image_id_map = build_image_id_map(pos_imgs, max_image_id(cvat_imgs) + 1)
    remapped_pos_imgs = [(image_id_map[old_id], image_b64) for old_id, image_b64 in pos_imgs]
    remapped_pos_texts = remap_pos_texts(pos_texts, image_id_map, max_text_id(cvat_texts) + 1)

    copy_cvat_non_train_files(cvat_dir, out_dir)
    write_jsonl(out_dir / "train_texts.jsonl", [*cvat_texts, *remapped_pos_texts])
    write_tsv(out_dir / "train_imgs.tsv", [*cvat_imgs, *remapped_pos_imgs])

    print(f"wrote output dir: {out_dir}")
    print(f"train texts: {len(cvat_texts)} + {len(remapped_pos_texts)} = {len(cvat_texts) + len(remapped_pos_texts)}")
    print(f"train images: {len(cvat_imgs)} + {len(remapped_pos_imgs)} = {len(cvat_imgs) + len(remapped_pos_imgs)}")
    if remapped_pos_texts:
        print(f"added text_id range: {remapped_pos_texts[0]['text_id']}..{remapped_pos_texts[-1]['text_id']}")
    if remapped_pos_imgs:
        print(f"added image_id range: {remapped_pos_imgs[0][0]}..{remapped_pos_imgs[-1][0]}")


if __name__ == "__main__":
    main()
