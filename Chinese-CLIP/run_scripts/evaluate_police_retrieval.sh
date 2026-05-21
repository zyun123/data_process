#!/usr/bin/env bash
set -euo pipefail

# Run from the Chinese-CLIP repo root, or from anywhere with REPO_DIR set.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-datapath/datasets/MUGE}"
CHECKPOINT="${1:-${CHECKPOINT:-}}"
TAG="${2:-${TAG:-}}"

VISION_MODEL="${VISION_MODEL:-ViT-B-16}"
TEXT_MODEL="${TEXT_MODEL:-RoBERTa-wwm-ext-base-chinese}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-52}"
IMG_BATCH_SIZE="${IMG_BATCH_SIZE:-64}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32768}"
EVAL_SPLIT="${EVAL_SPLIT:-valid}"

if [[ -z "${CHECKPOINT}" ]]; then
  cat >&2 <<USAGE
Usage:
  bash run_scripts/evaluate_police_retrieval.sh <checkpoint.pt> [tag]

Example:
  bash run_scripts/evaluate_police_retrieval.sh \\
    datapath/experiments/police_retrieval_hardneg_vit-b-16_bs8/checkpoints/epoch4.pt \\
    hardneg_bs8_epoch4

Optional env vars:
  PYTHON_BIN, DATASET_DIR, EVAL_SPLIT, VISION_MODEL, TEXT_MODEL, CONTEXT_LENGTH,
  IMG_BATCH_SIZE, TEXT_BATCH_SIZE, EVAL_BATCH_SIZE
USAGE
  exit 2
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ -z "${TAG}" ]]; then
  base="$(basename "${CHECKPOINT}")"
  exp="$(basename "$(dirname "$(dirname "${CHECKPOINT}")")")"
  TAG="${exp}_${base%.pt}"
fi

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

VALID_IMG_LMDB="${DATASET_DIR}/lmdb/${EVAL_SPLIT}/imgs"
VALID_TEXTS="${DATASET_DIR}/${EVAL_SPLIT}_texts.jsonl"
VALID_IMAGE_FEATS="${DATASET_DIR}/${EVAL_SPLIT}_${TAG}_imgs.img_feat.jsonl"
VALID_TEXT_FEATS="${DATASET_DIR}/${EVAL_SPLIT}_${TAG}_texts.txt_feat.jsonl"
VALID_PRED="${DATASET_DIR}/${EVAL_SPLIT}_predictions_${TAG}.jsonl"
VALID_EVAL="${DATASET_DIR}/${EVAL_SPLIT}_eval_${TAG}.json"
BUCKET_EVAL="${DATASET_DIR}/${EVAL_SPLIT}_bucket_eval_${TAG}.json"
BUCKET_CSV="${DATASET_DIR}/${EVAL_SPLIT}_bucket_eval_${TAG}.csv"

HARD_EVAL_SOURCE_SPLIT="${HARD_EVAL_SOURCE_SPLIT:-${EVAL_SPLIT}}"
HARD_SPLIT="${HARD_SPLIT:-${HARD_EVAL_SOURCE_SPLIT}_hard_eval}"
SOURCE_HARD_NEGATIVES="${DATASET_DIR}/${HARD_EVAL_SOURCE_SPLIT}_hard_negatives.jsonl"
HARD_IMG_LMDB="${DATASET_DIR}/lmdb/${HARD_SPLIT}/imgs"
HARD_TEXTS="${DATASET_DIR}/${HARD_SPLIT}_texts.jsonl"
HARD_META="${DATASET_DIR}/${HARD_SPLIT}_meta.jsonl"
HARD_IMAGE_FEATS="${DATASET_DIR}/${HARD_SPLIT}_${TAG}_imgs.img_feat.jsonl"
HARD_TEXT_FEATS="${DATASET_DIR}/${HARD_SPLIT}_${TAG}_texts.txt_feat.jsonl"
HARD_EVAL="${DATASET_DIR}/${HARD_SPLIT}_${TAG}.json"
HARD_MISSES="${DATASET_DIR}/${HARD_SPLIT}_${TAG}_misses.jsonl"

echo "[1/7] Check dataset files"
for path in "${VALID_IMG_LMDB}" "${VALID_TEXTS}" "${DATASET_DIR}/${EVAL_SPLIT}_imgs.tsv"; do
  if [[ ! -e "${path}" ]]; then
    echo "required file/directory not found: ${path}" >&2
    exit 1
  fi
done

echo "[2/7] Prepare hard-negative eval split if needed"
HARD_NEG_ROWS=0
if [[ -f "${SOURCE_HARD_NEGATIVES}" ]]; then
  HARD_NEG_ROWS="$(wc -l < "${SOURCE_HARD_NEGATIVES}")"
fi

if [[ "${HARD_NEG_ROWS}" -gt 0 && ( ! -f "${HARD_TEXTS}" || ! -f "${HARD_META}" || ! -f "${DATASET_DIR}/${HARD_SPLIT}_imgs.tsv" ) ]]; then
  "${PYTHON_BIN}" prepare_hard_negative_eval.py \
    --dataset-dir "${DATASET_DIR}" \
    --source-split "${HARD_EVAL_SOURCE_SPLIT}" \
    --out-split "${HARD_SPLIT}"
fi

if [[ "${HARD_NEG_ROWS}" -gt 0 && ! -d "${HARD_IMG_LMDB}" ]]; then
  "${PYTHON_BIN}" cn_clip/preprocess/build_lmdb_dataset.py \
    --data_dir "${DATASET_DIR}" \
    --splits "${HARD_SPLIT}"
fi

echo "[3/7] Extract ${EVAL_SPLIT} features"
"${PYTHON_BIN}" -u cn_clip/eval/extract_features.py \
  --extract-image-feats \
  --extract-text-feats \
  --image-data "${VALID_IMG_LMDB}" \
  --text-data "${VALID_TEXTS}" \
  --image-feat-output-path "${VALID_IMAGE_FEATS}" \
  --text-feat-output-path "${VALID_TEXT_FEATS}" \
  --resume "${CHECKPOINT}" \
  --vision-model "${VISION_MODEL}" \
  --text-model "${TEXT_MODEL}" \
  --context-length "${CONTEXT_LENGTH}" \
  --img-batch-size "${IMG_BATCH_SIZE}" \
  --text-batch-size "${TEXT_BATCH_SIZE}"

echo "[4/7] Evaluate ${EVAL_SPLIT} retrieval"
"${PYTHON_BIN}" -u cn_clip/eval/make_topk_predictions.py \
  --image-feats "${VALID_IMAGE_FEATS}" \
  --text-feats "${VALID_TEXT_FEATS}" \
  --top-k 10 \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --output "${VALID_PRED}"

"${PYTHON_BIN}" -u cn_clip/eval/evaluation.py \
  "${VALID_TEXTS}" \
  "${VALID_PRED}" \
  "${VALID_EVAL}"

echo "[5/7] Evaluate bucket retrieval"
"${PYTHON_BIN}" -u evaluate_bucket_retrieval.py \
  --texts "${VALID_TEXTS}" \
  --predictions "${VALID_PRED}" \
  --out "${BUCKET_EVAL}" \
  --csv-out "${BUCKET_CSV}"

if [[ "${HARD_NEG_ROWS}" -gt 0 ]]; then
  echo "[6/7] Extract hard-negative eval features (${HARD_EVAL_SOURCE_SPLIT})"
  "${PYTHON_BIN}" -u cn_clip/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data "${HARD_IMG_LMDB}" \
    --text-data "${HARD_TEXTS}" \
    --image-feat-output-path "${HARD_IMAGE_FEATS}" \
    --text-feat-output-path "${HARD_TEXT_FEATS}" \
    --resume "${CHECKPOINT}" \
    --vision-model "${VISION_MODEL}" \
    --text-model "${TEXT_MODEL}" \
    --context-length "${CONTEXT_LENGTH}" \
    --img-batch-size "${IMG_BATCH_SIZE}" \
    --text-batch-size "${TEXT_BATCH_SIZE}"

  echo "[7/7] Evaluate hard-negative ranking (${HARD_EVAL_SOURCE_SPLIT})"
  "${PYTHON_BIN}" -u evaluate_hard_negative_ranking.py \
    --meta "${HARD_META}" \
    --image-feats "${HARD_IMAGE_FEATS}" \
    --text-feats "${HARD_TEXT_FEATS}" \
    --out "${HARD_EVAL}" \
    --misses-out "${HARD_MISSES}"
else
  echo "[6/7] Skip hard-negative eval: ${SOURCE_HARD_NEGATIVES} is missing or empty"
  echo "[7/7] Skip hard-negative ranking"
fi

echo
echo "${EVAL_SPLIT} retrieval result:"
cat "${VALID_EVAL}"
echo
echo
echo "Hard-negative result:"
if [[ -f "${HARD_EVAL}" ]]; then
  cat "${HARD_EVAL}"
else
  echo "skipped: no ${HARD_EVAL_SOURCE_SPLIT} hard negatives"
fi
echo
echo
echo "Bucket result:"
cat "${BUCKET_CSV}"
echo
echo
echo "Outputs:"
echo "  ${VALID_EVAL}"
echo "  ${VALID_PRED}"
echo "  ${BUCKET_EVAL}"
echo "  ${BUCKET_CSV}"
if [[ -f "${HARD_EVAL}" ]]; then
  echo "  ${HARD_EVAL}"
  echo "  ${HARD_MISSES}"
fi
