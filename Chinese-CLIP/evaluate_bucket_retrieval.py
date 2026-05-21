#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Callable


COLORS = [
    "黑色",
    "白色",
    "红色",
    "蓝色",
    "绿色",
    "黄色",
    "灰色",
    "粉色",
    "紫色",
    "棕色",
    "橙色",
    "浅色",
    "深色",
    "黑",
    "白",
    "红",
    "蓝",
    "绿",
    "黄",
    "灰",
    "粉",
    "紫",
    "棕",
    "橙",
]

UPPER_TERMS = ["上衣", "外套", "短袖", "长袖", "衣服", "羽绒服", "羽绒", "夹克", "卫衣", "衬衫", "T恤", "校服"]
PANTS_TERMS = ["裤子", "长裤", "短裤", "裤"]
HELMET_TERMS = ["头盔", "帽子", "帽"]
BAG_TERMS = ["背包", "书包", "挎包", "包"]
EBIKE_TERMS = ["电动车", "电瓶车"]
MOTOR_TERMS = ["摩托车", "摩托"]
TRICYCLE_TERMS = ["三轮车", "三轮"]
VEHICLE_TERMS = EBIKE_TERMS + MOTOR_TERMS + TRICYCLE_TERMS + ["自行车", "车"]
WINDSHIELD_TERMS = ["挡风被", "挡风"]
PERSON_ROLE_TERMS = ["男子", "男人", "女士", "女人", "女子", "女孩", "男孩", "小孩", "儿童", "学生", "骑手", "乘客"]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def color_pattern() -> str:
    return "|".join(re.escape(color) for color in sorted(COLORS, key=len, reverse=True))


def term_pattern(terms: list[str]) -> str:
    return "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))


def has_colored_object(text: str, terms: list[str], window: int = 6) -> bool:
    colors = color_pattern()
    objects = term_pattern(terms)
    return bool(
        re.search(rf"({colors}).{{0,{window}}}({objects})", text)
        or re.search(rf"({objects}).{{0,{window}}}({colors})", text)
    )


def has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def has_multi_person_signal(text: str) -> bool:
    if "；" in text or ";" in text:
        return True
    signals = ["两人", "多人", "两个", "三人", "两名", "多名", "载人", "载着", "后座", "乘客", "带着小孩", "抱着孩子"]
    if any(signal in text for signal in signals):
        return True
    roles = re.findall(term_pattern(PERSON_ROLE_TERMS), text)
    return len(roles) >= 2


def read_reference(path: Path) -> dict[int, set[int]]:
    reference = {}
    for row in read_jsonl(path):
        reference[int(row["text_id"])] = {int(image_id) for image_id in row.get("image_ids", [])}
    return reference


def read_texts(path: Path) -> dict[int, str]:
    return {int(row["text_id"]): str(row.get("text", "")) for row in read_jsonl(path)}


def read_predictions(path: Path) -> dict[int, list[int]]:
    predictions = {}
    for row in read_jsonl(path):
        predictions[int(row["text_id"])] = [int(image_id) for image_id in row.get("image_ids", [])]
    return predictions


def recall_at(reference_ids: set[int], predicted_ids: list[int], k: int) -> bool:
    return any(image_id in reference_ids for image_id in predicted_ids[:k])


def bucket_defs() -> list[tuple[str, Callable[[str], bool]]]:
    return [
        ("upper_color", lambda text: has_colored_object(text, UPPER_TERMS)),
        ("pants_color", lambda text: has_colored_object(text, PANTS_TERMS)),
        ("helmet_or_hat_color", lambda text: has_colored_object(text, HELMET_TERMS)),
        ("bag_color", lambda text: has_colored_object(text, BAG_TERMS)),
        ("ebike_color", lambda text: has_colored_object(text, EBIKE_TERMS)),
        ("motor_ebike_tricycle_type", lambda text: has_any(text, EBIKE_TERMS) or has_any(text, MOTOR_TERMS) or has_any(text, TRICYCLE_TERMS)),
        ("windshield_color", lambda text: has_colored_object(text, WINDSHIELD_TERMS)),
        ("upper_vs_vehicle_color", lambda text: has_colored_object(text, UPPER_TERMS) and has_colored_object(text, VEHICLE_TERMS)),
        ("helmet_vs_upper_color", lambda text: has_colored_object(text, HELMET_TERMS) and has_colored_object(text, UPPER_TERMS)),
        ("multi_person_or_passenger", has_multi_person_signal),
    ]


def compute_bucket_metrics(
    texts: dict[int, str],
    reference: dict[int, set[int]],
    predictions: dict[int, list[int]],
) -> tuple[dict, list[dict]]:
    buckets = bucket_defs()
    rows = []
    for bucket_name, matcher in buckets:
        text_ids = [text_id for text_id, text in texts.items() if matcher(text)]
        if not text_ids:
            rows.append(
                {
                    "bucket": bucket_name,
                    "count": 0,
                    "r1": None,
                    "r5": None,
                    "r10": None,
                    "miss_r1": [],
                    "miss_r10": [],
                }
            )
            continue

        r1_hits = 0
        r5_hits = 0
        r10_hits = 0
        miss_r1 = []
        miss_r10 = []
        for text_id in text_ids:
            if text_id not in reference:
                raise ValueError(f"Missing reference for text_id={text_id}")
            if text_id not in predictions:
                raise ValueError(f"Missing prediction for text_id={text_id}")
            ref_ids = reference[text_id]
            pred_ids = predictions[text_id]
            hit1 = recall_at(ref_ids, pred_ids, 1)
            hit5 = recall_at(ref_ids, pred_ids, 5)
            hit10 = recall_at(ref_ids, pred_ids, 10)
            r1_hits += int(hit1)
            r5_hits += int(hit5)
            r10_hits += int(hit10)
            if not hit1:
                miss_r1.append({"text_id": text_id, "text": texts[text_id], "top1": pred_ids[:1]})
            if not hit10:
                miss_r10.append({"text_id": text_id, "text": texts[text_id], "top10": pred_ids[:10]})

        count = len(text_ids)
        rows.append(
            {
                "bucket": bucket_name,
                "count": count,
                "r1": r1_hits / count,
                "r5": r5_hits / count,
                "r10": r10_hits / count,
                "miss_r1": miss_r1,
                "miss_r10": miss_r10,
            }
        )

    overall_count = len(reference)
    overall = {
        "count": overall_count,
        "r1": sum(recall_at(reference[text_id], predictions[text_id], 1) for text_id in reference) / overall_count,
        "r5": sum(recall_at(reference[text_id], predictions[text_id], 5) for text_id in reference) / overall_count,
        "r10": sum(recall_at(reference[text_id], predictions[text_id], 10) for text_id in reference) / overall_count,
    }
    return overall, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bucket", "count", "r1", "r5", "r10", "miss_r1_count", "miss_r10_count"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "bucket": row["bucket"],
                    "count": row["count"],
                    "r1": "" if row["r1"] is None else round(row["r1"] * 100, 4),
                    "r5": "" if row["r5"] is None else round(row["r5"] * 100, 4),
                    "r10": "" if row["r10"] is None else round(row["r10"] * 100, 4),
                    "miss_r1_count": len(row["miss_r1"]),
                    "miss_r10_count": len(row["miss_r10"]),
                }
            )


def print_summary(overall: dict, rows: list[dict]) -> None:
    print("overall,count,r1,r5,r10")
    print(
        "overall,{},{:.4f},{:.4f},{:.4f}".format(
            overall["count"],
            overall["r1"] * 100,
            overall["r5"] * 100,
            overall["r10"] * 100,
        )
    )
    print("bucket,count,r1,r5,r10,miss_r1_count,miss_r10_count")
    for row in rows:
        r1 = "" if row["r1"] is None else f"{row['r1'] * 100:.4f}"
        r5 = "" if row["r5"] is None else f"{row['r5'] * 100:.4f}"
        r10 = "" if row["r10"] is None else f"{row['r10'] * 100:.4f}"
        print(
            f"{row['bucket']},{row['count']},{r1},{r5},{r10},"
            f"{len(row['miss_r1'])},{len(row['miss_r10'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics by police retrieval attribute buckets.")
    parser.add_argument("--texts", type=Path, required=True, help="MUGE *_texts.jsonl with ground-truth image_ids.")
    parser.add_argument("--predictions", type=Path, required=True, help="Top-k prediction jsonl.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON summary.")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional compact CSV output.")
    parser.add_argument("--print-full", action="store_true", help="Print full JSON, including miss details.")
    args = parser.parse_args()

    texts = read_texts(args.texts)
    reference = read_reference(args.texts)
    predictions = read_predictions(args.predictions)
    overall, rows = compute_bucket_metrics(texts, reference, predictions)

    result = {"overall": overall, "buckets": rows}
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    if args.csv_out:
        write_csv(args.csv_out, rows)

    if args.print_full:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_summary(overall, rows)
    print(f"wrote: {args.out}")
    if args.csv_out:
        print(f"wrote: {args.csv_out}")


if __name__ == "__main__":
    main()
