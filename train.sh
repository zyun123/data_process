
ln -s  custom datasets   MUGE


python cn_clip/preprocess/build_lmdb_dataset.py \
    --data_dir datapath/datasets/MUGE \
    --splits train,valid,valid_cvat_long


#------------------------------------------------------------------------------
cd /home/zy/Downloads/data_process/Chinese-CLIP


CUDA_VISIBLE_DEVICES=0,1,2,3 \
  VALID_BATCH_SIZE=8 \
  HARD_NEGATIVE_WEIGHT=0.05 \
  EXP_NAME=police_retrieval_cleaned_hardneg005_vit-b-16_g4_globalbs32 \
  bash run_scripts/custom_muge_finetune_1gpu.sh

如果这轮能稳定跑完，再试一轮全局 batch 64：

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  GPUS_PER_NODE=4 \
  BATCH_SIZE=16 \
  VALID_BATCH_SIZE=16 \
  HARD_NEGATIVE_WEIGHT=0.05 \
  EXP_NAME=police_retrieval_cleaned_hardneg005_vit-b-16_g4_globalbs64 \
  bash run_scripts/custom_muge_finetune_1gpu.sh


#最优
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  GPUS_PER_NODE=4 \
  BATCH_SIZE=8 \
  VALID_BATCH_SIZE=8 \
  HARD_NEGATIVE_WEIGHT=0.025 \
  EXP_NAME=0521_hardneg0025_g4_globalbs32_neg_rand_seed \
  bash run_scripts/custom_muge_finetune_1gpu.sh

# 和：

  # CUDA_VISIBLE_DEVICES=0,1,2,3 \
  # GPUS_PER_NODE=4 \
  # BATCH_SIZE=8 \
  # VALID_BATCH_SIZE=8 \
  # HARD_NEGATIVE_WEIGHT=0.1 \
  # EXP_NAME=police_retrieval_cleaned_hardneg01_vit-b-16_g4_globalbs32 \
  # bash run_scripts/custom_muge_finetune_1gpu.sh

#------------------------------------------------------------------------

# HARD_EVAL_SOURCE_SPLIT=train \
#   bash run_scripts/evaluate_police_retrieval.sh \
#     datapath/experiments/实验名/checkpoints/epoch4.pt \
#     实验名_epoch4_trainhard


bash run_scripts/evaluate_police_retrieval.sh \
    datapath/experiments/xxx/checkpoints/epoch5.pt \
    xxx_epoch5


EVAL_SPLIT=valid_cvat_long \
  bash run_scripts/evaluate_police_retrieval.sh \
    datapath/experiments/xxx/checkpoints/epoch5.pt \
    xxx_epoch5_cvatlong
  

#跑完评估后，把生成的 valid_eval_*.json 和 valid_bucket_eval_*.csv 拷回来，我再帮你定最终用哪个checkpoint。

对，主验证集还是拷这两个：

valid_eval_*.json
valid_bucket_eval_*.csv

但现在因为 valid_hard_negatives.jsonl 不为空了，还会额外生成 hard-negative 专项结果，也一起拷回来：

valid_hard_eval_*.json
valid_hard_eval_*_misses.jsonl

如果你也跑了 valid_cvat_long，再拷这两个：

valid_cvat_long_eval_*.json
valid_cvat_long_bucket_eval_*.csv

所以最完整要拷：

cp valid_eval_*.json final_eval/
cp valid_bucket_eval_*.csv final_eval/
cp valid_hard_eval_*.json final_eval/
cp valid_hard_eval_*_misses.jsonl final_eval/

cp valid_cvat_long_eval_*.json final_eval/
cp valid_cvat_long_bucket_eval_*.csv final_eval/











CUDA_VISIBLE_DEVICES=0,1,2,3 \
  GPUS_PER_NODE=4 \
  BATCH_SIZE=32 \
  VALID_BATCH_SIZE=32 \
  HARD_NEGATIVE_WEIGHT=0.025 \
  EXP_NAME=0526_bs32 \
  bash run_scripts/custom_muge_finetune_1gpu_epoch_50.sh






MAX_EPOCHS=10 LR=1e-6 WD=0.001 CUDA_VISIBLE_DEVICES=0,1,2,3 \


CUDA_VISIBLE_DEVICES=0,1,2,3 \
  MAX_EPOCHS=30 \
  LR=3e-6 \
  WD=0.01 \
  GPUS_PER_NODE=4 \
  BATCH_SIZE=16 \
  VALID_BATCH_SIZE=16 \
  HARD_NEGATIVE_WEIGHT=0.025 \
  EXP_NAME=0527_bs16_lr3e-6_wd001 \
  bash run_scripts/custom_muge_finetune_4gpu_epoch20.sh