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

if [[ -e "${OUT_ROOT}" ]]; then
  echo "Output directory already exists: ${OUT_ROOT}" >&2
  echo "Use a new directory, for example: ./处理流程.sh datasets/$(date +%F)-processed-v2" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
echo "Output root: ${OUT_ROOT}"

# 1. CVAT COCO 数据集 -> MUGE 数据集。
python3 cvat_to_muge.py \
  --inputs \
    cvat_data/task_2026-04-29 \
    cvat_data/task_2026-04-29-2 \
    cvat_data/task_2026-05-07 \
    cvat_data/task_2026-05-11 \
  --output-dir "${CVAT_DATASET}"

# 2. 过滤过长文本，并同步过滤不再被引用的图片。
python3 filter_long_texts.py \
  --dataset-dir "${CVAT_DATASET}" \
  --out-dir "${CVAT_NO_LONG_DATASET}" \
  --max-len 55

# 3. 从 pos/neg 采集数据中抽取所有 pos_images，并按 query 目录切分 train/valid。
# 后续 merge_pos_to_cvat.py 会重新映射 ID，所以这里从 0 开始即可。
python3 build_muge_all_pos.py \
  --root /home/zy/Downloads/chinese_clip/pos_and_neg_datasets/ \
  --out-dir "${POS_DATASET}" \
  --text-start-id 0 \
  --image-start-id 0 \
  --valid-ratio 0.1 \
  --seed 42

# 4. 将 pos 数据集合并到过滤后的 CVAT 数据集。
python3 merge_pos_to_cvat.py \
  --cvat-dir "${CVAT_NO_LONG_DATASET}" \
  --pos-dir "${POS_DATASET}" \
  --out-dir "${MERGED_DATASET}"

# 5. 扩展 train_texts.jsonl，输出完整 MUGE 数据集目录。
# python3 expand_train_texts.py \
#   --input "${MERGED_DATASET}" \
#   --output "${EXPANDED_DATASET}" \
#   --max-new-per-item 4

python expand_text_variants.py \
  --dataset-dir "${MERGED_DATASET}" \
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
