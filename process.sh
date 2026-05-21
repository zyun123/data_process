#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-datasets/$(date +%F)-processed}"
# OUT_ROOT="datasets/0518_nosplit"
CVAT_DATASET="${OUT_ROOT}/cvat_datasets"
CVAT_NO_LONG_DATASET="${OUT_ROOT}/cvat_no_long_datasets"
POS_DATASET="${OUT_ROOT}/pos_datasets"
MERGED_DATASET="${OUT_ROOT}/cvat_merge_pos_datasets"
# EXPANDED_DATASET="${OUT_ROOT}/cvat_merge_pos_datasets_expanded_$(date +%F)"
EXPANDED_DATASET="${OUT_ROOT}/cvat_merge_pos_datasets_expand"
POS_NEG_ROOT="${POS_NEG_ROOT:-/home/zy/Downloads/chinese_clip/pos_and_neg_datasets/}"
EXTRA_POS_NEG_ROOT="${EXTRA_POS_NEG_ROOT:-collection_rounds/2026-05-19_priority3_round1}"
EXTRA_POS_DATASET="${OUT_ROOT}/extra_pos_datasets"
FINAL_MERGED_DATASET="${MERGED_DATASET}"
CLEANED_DATASET="${OUT_ROOT}/cvat_merge_cleaned_datasets"

if [[ -e "${OUT_ROOT}" ]]; then
  echo "Output directory already exists: ${OUT_ROOT}" >&2
  echo "Use a new directory, for example: ./process.sh datasets/$(date +%F)-processed-v2" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
echo "Output root: ${OUT_ROOT}"
echo "Pos/neg root: ${POS_NEG_ROOT}"
echo "Extra pos/neg root: ${EXTRA_POS_NEG_ROOT:-<none>}"

# 1. CVAT COCO 数据集 -> MUGE 数据集。
python3 cvat_to_muge.py \
  --inputs \
    cvat_data/task_2026-04-29 \
    cvat_data/task_2026-04-29-2 \
    cvat_data/task_2026-05-07 \
    cvat_data/task_2026-05-11 \
  --text-mode boxes \
  --output-dir "${CVAT_DATASET}" \
  --valid-ratio 0.4

# 2. 过滤过长文本，并同步过滤不再被引用的图片。
python3 filter_long_texts.py \
  --dataset-dir "${CVAT_DATASET}" \
  --out-dir "${CVAT_NO_LONG_DATASET}" \
  --max-len 55

# 3. 从 pos/neg 采集数据中抽取所有 pos_images，并保留 neg_images 作为 hard-negative 元数据。
#    按 query 目录切分 train/valid，避免同一 query 的正图跨 split 泄漏。
# 后续 merge_pos_to_cvat.py 会重新映射 ID，所以这里从 0 开始即可。
python3 build_muge_all_pos.py \
  --root "${POS_NEG_ROOT}" \
  --out-dir "${POS_DATASET}" \
  --text-start-id 0 \
  --image-start-id 0 \
  --valid-ratio 0.0 \
  --seed 42

# 4. 将 pos 数据集合并到过滤后的 CVAT 数据集。
python3 merge_pos_to_cvat.py \
  --cvat-dir "${CVAT_NO_LONG_DATASET}" \
  --pos-dir "${POS_DATASET}" \
  --out-dir "${MERGED_DATASET}"

if [[ -n "${EXTRA_POS_NEG_ROOT}" && -d "${EXTRA_POS_NEG_ROOT}" ]]; then
  python3 build_muge_all_pos.py \
    --root "${EXTRA_POS_NEG_ROOT}" \
    --out-dir "${EXTRA_POS_DATASET}" \
    --text-start-id 0 \
    --image-start-id 0 \
    --valid-ratio 0.0 \
    --seed 42

  FINAL_MERGED_DATASET="${OUT_ROOT}/cvat_merge_extra_pos_datasets"
  python3 merge_pos_to_cvat.py \
    --cvat-dir "${MERGED_DATASET}" \
    --pos-dir "${EXTRA_POS_DATASET}" \
    --out-dir "${FINAL_MERGED_DATASET}"
else
  echo "Skip extra pos/neg root: ${EXTRA_POS_NEG_ROOT:-<none>}"
fi

# 5. 清洗已知 query 噪声，避免错别字和脏文本被后续扩充放大。
python3 clean_query_texts.py \
  --dataset-dir "${FINAL_MERGED_DATASET}" \
  --out-dir "${CLEANED_DATASET}" \
  --report-out "${OUT_ROOT}/query_cleaning_report.jsonl"

# 6. 扩展 train_texts.jsonl，输出完整 MUGE 数据集目录。
# python3 expand_train_texts.py \
#   --input "${MERGED_DATASET}" \
#   --output "${EXPANDED_DATASET}" \
#   --max-new-per-item 4

python expand_text_variants.py \
  --dataset-dir "${CLEANED_DATASET}" \
  --out-dir "${EXPANDED_DATASET}" \
  --split train \
  --max-new 2 \
  --min-len 6 \
  --max-len 45 \
  --single-image-only

python3 validate_muge_dataset.py \
  --dataset-dir "${EXPANDED_DATASET}"

echo "Done."
echo "Final dataset: ${EXPANDED_DATASET}"
