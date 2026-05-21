#!/usr/bin/env python3
import argparse
import csv
import json
import re
from datetime import date
from pathlib import Path


TASKS = [
    {
        "priority": "P1",
        "bucket": "helmet_vs_vehicle_color",
        "query": "戴黑色头盔穿白色上衣黑色裤子骑红色电动车的人",
        "pos_min": 30,
        "neg_min": 30,
        "why": "hard negative 曾在黑头盔、白上衣、红电动车组合上失败；需要学习头盔/上衣/车体颜色绑定。",
        "neg_queries": [
            "戴黑色头盔穿白色上衣黑色裤子骑黑色电动车的人",
            "戴白色头盔穿白色上衣黑色裤子骑红色电动车的人",
            "戴黑色头盔穿红色上衣黑色裤子骑白色电动车的人",
            "红色电动车在旁边但人没有骑的人",
        ],
    },
    {
        "priority": "P1",
        "bucket": "vehicle_type",
        "query": "骑黑色电动车的人",
        "pos_min": 40,
        "neg_min": 40,
        "why": "车型混淆是当前核心短板；黑色电动车容易和黑色摩托车、黑色三轮车混。",
        "neg_queries": [
            "骑黑色摩托车的人",
            "骑黑色三轮车的人",
            "骑黑色自行车的人",
            "站在黑色电动车旁边但没有骑的人",
        ],
    },
    {
        "priority": "P1",
        "bucket": "vehicle_type",
        "query": "骑黑色摩托车的人",
        "pos_min": 40,
        "neg_min": 40,
        "why": "和“骑黑色电动车的人”成对采集，训练模型区分摩托车/电动车结构。",
        "neg_queries": [
            "骑黑色电动车的人",
            "骑黑色三轮车的人",
            "推黑色摩托车的人",
            "黑色摩托车在旁边但人没有骑的人",
        ],
    },
    {
        "priority": "P1",
        "bucket": "upper_pants_binding",
        "query": "穿黑色短袖白色裤子的人",
        "pos_min": 40,
        "neg_min": 40,
        "why": "hard negative 仍失败；需要加强黑上衣/白裤子的上下装绑定。",
        "neg_queries": [
            "穿白色短袖黑色裤子的人",
            "穿黑色长袖白色裤子的人",
            "穿黑色短袖黑色裤子的人",
            "穿白色上衣白色裤子的人",
        ],
    },
    {
        "priority": "P1",
        "bucket": "hat_color",
        "query": "戴黄色帽子的人",
        "pos_min": 30,
        "neg_min": 30,
        "why": "hard negative 仍失败；黄色帽子容易和黄色上衣、黄色头发、背景黄色物体混。",
        "neg_queries": [
            "穿黄色上衣但没有黄色帽子的人",
            "黄色头发但没有黄色帽子的人",
            "戴白色帽子穿黄色上衣的人",
            "黄色物体在旁边但人没有戴黄色帽子的人",
        ],
    },
    {
        "priority": "P2",
        "bucket": "upper_vs_vehicle_color",
        "query": "穿黑色上衣骑红色电动车的人",
        "pos_min": 30,
        "neg_min": 30,
        "why": "上衣颜色和车体颜色绑定是当前 bucket 短板。",
        "neg_queries": [
            "穿红色上衣骑黑色电动车的人",
            "穿黑色上衣骑黑色电动车的人",
            "穿红色上衣骑红色电动车的人",
            "红色电动车在旁边但人穿黑色上衣未骑车",
        ],
    },
    {
        "priority": "P2",
        "bucket": "upper_vs_vehicle_color",
        "query": "穿白色上衣骑黑色电动车的人",
        "pos_min": 30,
        "neg_min": 30,
        "why": "补充白上衣/黑车体的反向组合，降低颜色错绑。",
        "neg_queries": [
            "穿黑色上衣骑白色电动车的人",
            "穿白色上衣骑白色电动车的人",
            "穿黑色上衣骑黑色电动车的人",
            "白色上衣的人站在黑色电动车旁边",
        ],
    },
    {
        "priority": "P2",
        "bucket": "helmet_vs_upper_color",
        "query": "戴白色头盔穿蓝色上衣的人",
        "pos_min": 25,
        "neg_min": 25,
        "why": "头盔颜色和上衣颜色绑定需要成对补样。",
        "neg_queries": [
            "戴蓝色头盔穿白色上衣的人",
            "戴白色头盔穿白色上衣的人",
            "戴黑色头盔穿蓝色上衣的人",
            "蓝色上衣的人旁边有白色头盔但没有戴",
        ],
    },
    {
        "priority": "P3",
        "bucket": "upper_pants_binding",
        "query": "穿黑色上衣白色裤子的人",
        "pos_min": 40,
        "neg_min": 40,
        "why": "普通 valid 中黑上衣白裤子仍有大量 Top1 miss；需要和反向组合成对采集。",
        "neg_queries": [
            "穿白色上衣黑色裤子的人",
            "穿黑色上衣黑色裤子的人",
            "穿白色上衣白色裤子的人",
            "穿黑色裤子但上衣不是黑色的人",
        ],
    },
    {
        "priority": "P3",
        "bucket": "upper_pants_binding",
        "query": "穿白色上衣黑色裤子的人",
        "pos_min": 40,
        "neg_min": 40,
        "why": "和黑上衣白裤子互为 hard negative，专门提升上下装颜色绑定。",
        "neg_queries": [
            "穿黑色上衣白色裤子的人",
            "穿白色上衣白色裤子的人",
            "穿黑色上衣黑色裤子的人",
            "穿白色裤子但上衣不是白色的人",
        ],
    },
    {
        "priority": "P4",
        "bucket": "windshield_vs_vehicle_color",
        "query": "骑黑色电动车带白色挡风被的人",
        "pos_min": 25,
        "neg_min": 25,
        "why": "挡风被颜色和车体颜色容易互相污染，需要绑定采集。",
        "neg_queries": [
            "骑白色电动车带黑色挡风被的人",
            "骑黑色电动车带黑色挡风被的人",
            "骑白色电动车带白色挡风被的人",
            "白色挡风被在旁边但不在黑色电动车上",
        ],
    },
    {
        "priority": "P5",
        "bucket": "multi_person_or_passenger",
        "query": "骑电动车后座载人的人",
        "pos_min": 30,
        "neg_min": 30,
        "why": "多人/载人 query 容易选错主体；要补主体明确的正负样本。",
        "neg_queries": [
            "单人骑电动车的人",
            "两个人站在电动车旁边但没有载人",
            "骑电动车前面坐小孩的人",
            "后座有物品但没有人的电动车",
        ],
    },
    {
        "priority": "P6",
        "bucket": "bag_color",
        "query": "背黑色背包穿白色上衣的人",
        "pos_min": 25,
        "neg_min": 25,
        "why": "bag_color 的 R@1 当前最低，需要补包颜色和上衣颜色绑定。",
        "neg_queries": [
            "背白色背包穿黑色上衣的人",
            "背黑色背包穿黑色上衣的人",
            "背红色背包穿白色上衣的人",
            "穿白色上衣但没有背黑色背包的人",
        ],
    },
]


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_name(text: str, max_len: int = 48) -> str:
    text = re.sub(r"[\\/:*?\"<>|\\s]+", "_", text.strip())
    return text[:max_len].strip("_")


def bucket_metrics(bucket_eval: dict | None) -> dict[str, dict]:
    if not bucket_eval:
        return {}
    return {row["bucket"]: row for row in bucket_eval.get("buckets", [])}


def related_failures(task: dict, bucket_rows: dict[str, dict], hard_misses: list[dict], limit: int = 8) -> list[dict]:
    failures = []
    row = bucket_rows.get(task["bucket"])
    if row:
        for miss in row.get("miss_r10", [])[:limit]:
            failures.append({"type": "bucket_miss_r10", "text_id": miss["text_id"], "text": miss["text"]})
        if len(failures) < limit:
            for miss in row.get("miss_r1", [])[: limit - len(failures)]:
                failures.append({"type": "bucket_miss_r1", "text_id": miss["text_id"], "text": miss["text"]})

    query = task["query"]
    for miss in hard_misses:
        text = miss.get("text", "")
        if text in query or query in text:
            failures.insert(0, {"type": "hard_negative_top1", "text_id": miss.get("text_id"), "text": text})
    return failures[:limit]


def write_task_dir(root: Path, idx: int, task: dict, failures: list[dict]) -> Path:
    task_dir = root / f"{task['priority']}_{idx:03d}_{safe_name(task['query'])}"
    pos_dir = task_dir / "pos_images"
    neg_dir = task_dir / "neg_images"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)
    (pos_dir / ".gitkeep").touch()
    (neg_dir / ".gitkeep").touch()
    (task_dir / "query.txt").write_text(task["query"] + "\n", encoding="utf-8")
    (task_dir / "neg_queries.txt").write_text("\n".join(task["neg_queries"]) + "\n", encoding="utf-8")
    (task_dir / "source_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = f"""# {task['query']}

priority: {task['priority']}
bucket: {task['bucket']}
target positives: >= {task['pos_min']}
target hard negatives: >= {task['neg_min']}

why:
{task['why']}

positive rule:
- `pos_images/` 只放严格匹配 query 的图片。
- 主体要清楚，颜色/车型/上下装/头盔/包等关键属性必须能看出来。
- 如果是多人图，只放 query 主体明确的图。

hard negative rule:
- `neg_images/` 放视觉上很像但关键属性错误的图片。
- 优先按 `neg_queries.txt` 采集。
- 不要放完全无关图片；负样本要“像到模型容易错”。

notes:
- 每个目录的 `query.txt` 保持一行 query。
- 采集完后继续走现有 `process.sh`，这些目录会被 `build_muge_all_pos.py` 读入。
"""
    (task_dir / "README.md").write_text(readme, encoding="utf-8")
    return task_dir


def write_plan_csv(path: Path, task_rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "bucket",
                "query",
                "pos_min",
                "neg_min",
                "task_dir",
                "why",
                "neg_queries",
            ],
        )
        writer.writeheader()
        writer.writerows(task_rows)


def write_plan_md(path: Path, task_rows: list[dict], bucket_rows: dict[str, dict], hard_misses: list[dict]) -> None:
    lines = [
        "# Collection Round",
        "",
        "## Current Weak Buckets",
        "",
        "| bucket | count | r1 | r5 | r10 | miss_r10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bucket, row in sorted(bucket_rows.items(), key=lambda item: (item[1].get("r1") is None, item[1].get("r1") or 0)):
        r1 = "" if row.get("r1") is None else f"{row['r1'] * 100:.2f}"
        r5 = "" if row.get("r5") is None else f"{row['r5'] * 100:.2f}"
        r10 = "" if row.get("r10") is None else f"{row['r10'] * 100:.2f}"
        lines.append(f"| {bucket} | {row.get('count', 0)} | {r1} | {r5} | {r10} | {len(row.get('miss_r10', []))} |")

    lines.extend(["", "## Hard Negative Top1 Misses", ""])
    if hard_misses:
        for row in hard_misses:
            lines.append(f"- text_id={row.get('text_id')}: {row.get('text')}")
    else:
        lines.append("- none")

    lines.extend(["", "## Tasks", ""])
    for row in task_rows:
        lines.extend(
            [
                f"### {row['priority']} {row['query']}",
                "",
                f"- bucket: `{row['bucket']}`",
                f"- target: pos >= {row['pos_min']}, hard neg >= {row['neg_min']}",
                f"- dir: `{row['task_dir']}`",
                f"- why: {row['why']}",
                f"- hard negative queries: {row['neg_queries']}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate failure-driven collection tasks for police text-to-image retrieval.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Chinese-CLIP/datapath/datasets/MUGE"))
    parser.add_argument("--tag", default="hardneg_bs8_epoch4", help="Evaluation tag used in valid_bucket_eval_<tag>.json.")
    parser.add_argument("--out-root", type=Path, default=Path("collection_rounds"))
    parser.add_argument("--round-name", default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    round_name = args.round_name or f"{date.today().isoformat()}_{args.tag}"
    out_dir = args.out_root / round_name
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_eval = read_json(dataset_dir / f"valid_bucket_eval_{args.tag}.json")
    hard_misses = read_jsonl(dataset_dir / f"valid_hard_eval_{args.tag}_misses.jsonl")
    bucket_rows = bucket_metrics(bucket_eval)

    task_rows = []
    for idx, task in enumerate(TASKS, start=1):
        failures = related_failures(task, bucket_rows, hard_misses)
        task_dir = write_task_dir(out_dir, idx, task, failures)
        task_rows.append(
            {
                "priority": task["priority"],
                "bucket": task["bucket"],
                "query": task["query"],
                "pos_min": task["pos_min"],
                "neg_min": task["neg_min"],
                "task_dir": str(task_dir),
                "why": task["why"],
                "neg_queries": " | ".join(task["neg_queries"]),
            }
        )

    write_plan_csv(out_dir / "collection_plan.csv", task_rows)
    write_plan_md(out_dir / "README.md", task_rows, bucket_rows, hard_misses)
    print(f"wrote: {out_dir}")
    print(f"tasks: {len(task_rows)}")
    print(f"plan: {out_dir / 'collection_plan.csv'}")
    print(f"readme: {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
