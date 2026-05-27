#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_scripts/custom_muge_finetune_1gpu.sh
#   bash run_scripts/custom_muge_finetune_1gpu.sh /path/to/datapath
#
# Expected layout:
#   ${DATAPATH}/datasets/MUGE/lmdb/train
#   ${DATAPATH}/datasets/MUGE/lmdb/valid
#   ${DATAPATH}/pretrained_weights/clip_cn_vit-b-16.pt

DATAPATH="${1:-$(pwd)/datapath}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-8514}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-${BATCH_SIZE}}"

MAX_EPOCHS="${MAX_EPOCHS:-20}"
LR="${LR:-3e-6}"
WD="${WD:-0.01}"


export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export RANK="${RANK:-0}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/cn_clip"

train_data="${DATAPATH}/datasets/MUGE/lmdb/train"
val_data="${DATAPATH}/datasets/MUGE/lmdb/valid"
resume="${DATAPATH}/pretrained_weights/clip_cn_vit-b-16.pt"
output_base_dir="${DATAPATH}/experiments"
name="${EXP_NAME:-police_retrieval_hardneg_vit-b-16_bs8}"

if [[ ! -d "${train_data}" ]]; then
  echo "Missing train LMDB: ${train_data}" >&2
  exit 1
fi
if [[ ! -d "${val_data}" ]]; then
  echo "Missing valid LMDB: ${val_data}" >&2
  exit 1
fi
if [[ ! -f "${resume}" ]]; then
  echo "Missing pretrained checkpoint: ${resume}" >&2
  echo "Put clip_cn_vit-b-16.pt under ${DATAPATH}/pretrained_weights/ first." >&2
  exit 1
fi

torchrun \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -- \
  cn_clip/training/main.py \
  --train-data="${train_data}" \
  --val-data="${val_data}" \
  --num-workers=4 \
  --valid-num-workers=2 \
  --resume="${resume}" \
  --reset-data-offset \
  --reset-optimizer \
  --logs="${output_base_dir}" \
  --name="${name}" \
  --save-step-frequency=999999 \
  --save-epoch-frequency=1 \
  --log-interval=10 \
  --report-training-batch-acc \
  --context-length=52 \
  --warmup=50 \
  --batch-size="${BATCH_SIZE}" \
  --valid-batch-size="${VALID_BATCH_SIZE}" \
  --valid-step-interval=50 \
  --valid-epoch-interval=1 \
  --accum-freq=1 \
  --lr="${LR}" \
  --wd="${WD}" \
  --max-epochs="${MAX_EPOCHS}" \
  --hard-negative-weight="${HARD_NEGATIVE_WEIGHT:-0.2}" \
  --vision-model=ViT-B-16 \
  --text-model=RoBERTa-wwm-ext-base-chinese \
  --use-augment \
  # --grad-checkpointing
