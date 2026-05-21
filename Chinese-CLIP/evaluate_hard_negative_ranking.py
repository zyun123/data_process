#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_features(path: Path, id_field: str) -> dict[int, np.ndarray]:
    feats = {}
    for row in read_jsonl(path):
        item_id = int(row[id_field])
        feat = np.asarray(row["feature"], dtype=np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        feats[item_id] = feat
    return feats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate positive ranks against per-query hard negatives.")
    parser.add_argument("--meta", type=Path, default=Path("datapath/datasets/MUGE/valid_hard_eval_meta.jsonl"))
    parser.add_argument("--image-feats", type=Path, default=Path("datapath/datasets/MUGE/valid_hard_eval_imgs.img_feat.jsonl"))
    parser.add_argument("--text-feats", type=Path, default=Path("datapath/datasets/MUGE/valid_hard_eval_texts.txt_feat.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("datapath/datasets/MUGE/valid_hard_eval.json"))
    parser.add_argument("--misses-out", type=Path, default=Path("datapath/datasets/MUGE/valid_hard_eval_misses.jsonl"))
    args = parser.parse_args()

    meta_rows = read_jsonl(args.meta)
    image_feats = load_features(args.image_feats, "image_id")
    text_feats = load_features(args.text_feats, "text_id")

    evaluated = []
    misses = []
    for row in meta_rows:
        text_id = int(row["text_id"])
        if text_id not in text_feats:
            raise ValueError(f"Missing text feature for text_id={text_id}")
        text_feat = text_feats[text_id]
        pos_ids = [int(image_id) for image_id in row["positive_image_ids"]]
        neg_ids = [int(image_id) for image_id in row["negative_image_ids"]]
        candidate_ids = pos_ids + neg_ids
        missing_images = [image_id for image_id in candidate_ids if image_id not in image_feats]
        if missing_images:
            raise ValueError(f"text_id={text_id} missing image features: {missing_images[:10]}")

        scored = []
        for image_id in candidate_ids:
            score = float(text_feat @ image_feats[image_id])
            scored.append((image_id, score, image_id in pos_ids))
        scored.sort(key=lambda item: item[1], reverse=True)

        best_pos_rank = None
        best_neg_above = 0
        for rank, (_, _, is_pos) in enumerate(scored, start=1):
            if is_pos:
                best_pos_rank = rank
                break
            best_neg_above += 1
        if best_pos_rank is None:
            raise ValueError(f"text_id={text_id} has no positive candidate")

        top1_positive = scored[0][2]
        top5_positive = any(is_pos for _, _, is_pos in scored[:5])
        top10_positive = any(is_pos for _, _, is_pos in scored[:10])
        top1_is_hard_negative = not top1_positive

        detail = {
            "text_id": text_id,
            "text": row["text"],
            "positive_image_ids": pos_ids,
            "negative_image_ids": neg_ids,
            "candidate_count": len(candidate_ids),
            "positive_count": len(pos_ids),
            "negative_count": len(neg_ids),
            "best_positive_rank": best_pos_rank,
            "hard_negatives_above_best_positive": best_neg_above,
            "top1_positive": top1_positive,
            "top5_positive": top5_positive,
            "top10_positive": top10_positive,
            "top10": [
                {"image_id": image_id, "score": score, "label": "positive" if is_pos else "hard_negative"}
                for image_id, score, is_pos in scored[:10]
            ],
        }
        evaluated.append(detail)
        if top1_is_hard_negative:
            misses.append(detail)

    total = len(evaluated)
    top1 = sum(item["top1_positive"] for item in evaluated) / total if total else 0.0
    top5 = sum(item["top5_positive"] for item in evaluated) / total if total else 0.0
    top10 = sum(item["top10_positive"] for item in evaluated) / total if total else 0.0
    mean_rank = sum(item["best_positive_rank"] for item in evaluated) / total if total else 0.0
    mean_hard_negs_above = (
        sum(item["hard_negatives_above_best_positive"] for item in evaluated) / total if total else 0.0
    )
    ranks = sorted(item["best_positive_rank"] for item in evaluated)
    median_rank = ranks[len(ranks) // 2] if ranks else 0

    result = {
        "total_queries": total,
        "top1_positive_rate": top1,
        "top5_positive_rate": top5,
        "top10_positive_rate": top10,
        "mean_best_positive_rank": mean_rank,
        "median_best_positive_rank": median_rank,
        "mean_hard_negatives_above_best_positive": mean_hard_negs_above,
        "top1_hard_negative_queries": len(misses),
    }
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with args.misses_out.open("w", encoding="utf-8") as f:
        for item in misses:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote: {args.out}")
    print(f"wrote: {args.misses_out}")


if __name__ == "__main__":
    main()
