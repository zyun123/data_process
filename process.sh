#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-datasets/$(date +%F)-processed}"
# OUT_ROOT="datasets/0518_nosplit"
CVAT_DATASET="${OUT_ROOT}/cvat_datasets"
CVAT_NO_LONG_DATASET="${OUT_ROOT}/cvat_no_long_datasets"
POS_DATASET="${OUT_ROOT}/pos_datasets"
# EXPANDED_DATASET="${OUT_ROOT}/cvat_merge_pos_datasets_expanded_$(date +%F)"
EXPANDED_DATASET="${OUT_ROOT}/cvat_merge_pos_datasets_expand"
POS_NEG_ROOT="${POS_NEG_ROOT:-/home/zy/Downloads/chinese_clip/pos_and_neg_datasets/}"
EXTRA_POS_NEG_ROOT="${EXTRA_POS_NEG_ROOT:-}"
COLLECTION_ROOT="${COLLECTION_ROOT:-collection_rounds}"
POS_VALID_RATIO="${POS_VALID_RATIO:-0.15}"
FINAL_MERGED_DATASET="${OUT_ROOT}/cvat_pos_stratified_datasets"
CLEANED_DATASET="${OUT_ROOT}/cvat_merge_cleaned_datasets"

if [[ -e "${OUT_ROOT}" ]]; then
  echo "Output directory already exists: ${OUT_ROOT}" >&2
  echo "Use a new directory, for example: ./process.sh datasets/$(date +%F)-processed-v2" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
echo "Output root: ${OUT_ROOT}"
echo "Pos/neg root: ${POS_NEG_ROOT}"
echo "Train-only extra pos/neg root: ${EXTRA_POS_NEG_ROOT:-<auto from ${COLLECTION_ROOT}>}"
echo "Pos valid ratio: ${POS_VALID_RATIO}"

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

POS_ROOT_ARGS=("${POS_NEG_ROOT}")
TRAIN_ONLY_ROOT_ARGS=()
if [[ -n "${EXTRA_POS_NEG_ROOT}" && -d "${EXTRA_POS_NEG_ROOT}" ]]; then
  TRAIN_ONLY_ROOT_ARGS+=("${EXTRA_POS_NEG_ROOT}")
elif [[ -n "${EXTRA_POS_NEG_ROOT}" ]]; then
  echo "Skip extra pos/neg root: ${EXTRA_POS_NEG_ROOT}"
else
  while IFS= read -r round_dir; do
    TRAIN_ONLY_ROOT_ARGS+=("${round_dir}")
  done < <(
    find "${COLLECTION_ROOT}" -mindepth 1 -maxdepth 1 -type d \
      ! -name '*_output' \
      | sort
  )
fi
printf 'Use pos/neg roots:\n'
printf '  %s\n' "${POS_ROOT_ARGS[@]}"
if [[ "${#TRAIN_ONLY_ROOT_ARGS[@]}" -gt 0 ]]; then
  printf 'Use train-only pos/neg roots:\n'
  printf '  %s\n' "${TRAIN_ONLY_ROOT_ARGS[@]}"
fi

# 3. 从 pos/neg 采集数据中抽取正例和 hard negatives。
#    按 query 目录做多标签分层切分，避免按图片切分泄漏，同时让衣服/裤子/头盔/鞋/挡风被/车型等桶都有覆盖。
BUILD_POS_ARGS=(
  --root "${POS_ROOT_ARGS[@]}" \
  --out-dir "${POS_DATASET}" \
  --text-start-id 0 \
  --image-start-id 0 \
  --valid-ratio "${POS_VALID_RATIO}" \
  --split-strategy stratified \
  --min-train-per-label 1 \
  --seed 42
)
if [[ "${#TRAIN_ONLY_ROOT_ARGS[@]}" -gt 0 ]]; then
  BUILD_POS_ARGS+=(--train-root "${TRAIN_ONLY_ROOT_ARGS[@]}")
fi
python3 build_muge_all_pos.py "${BUILD_POS_ARGS[@]}"

# 4. 组合最终数据集：
#    train = CVAT train + pos stratified train
#    valid = pos stratified holdout，作为训练中的主验证集
#    valid_cvat_long = CVAT valid，作为长描述辅助评估集
python3 compose_police_retrieval_dataset.py \
  --cvat-dir "${CVAT_NO_LONG_DATASET}" \
  --pos-dir "${POS_DATASET}" \
  --out-dir "${FINAL_MERGED_DATASET}"

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
