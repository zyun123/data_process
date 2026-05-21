cd /home/zy/Downloads/data_process/Chinese-CLIP
bash run_scripts/custom_muge_finetune_1gpu.sh

python -u cn_clip/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="datapath/datasets/MUGE/lmdb/valid/imgs" \
    --text-data="datapath/datasets/MUGE/valid_texts.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=52 \
    --resume="datapath/experiments/police_retrieval_vit-b-16_bs32/checkpoints/epoch4.pt" \
    --vision-model=ViT-B-16 \
    --text-model=RoBERTa-wwm-ext-base-chinese




python -u cn_clip/eval/make_topk_predictions.py \
    --image-feats="datapath/datasets/MUGE/valid_imgs.img_feat.jsonl" \
    --text-feats="datapath/datasets/MUGE/valid_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="datapath/datasets/MUGE/valid_predictions.jsonl"


python cn_clip/eval/evaluation.py \
    datapath/datasets/MUGE/valid_texts.jsonl \
    datapath/datasets/MUGE/valid_predictions.jsonl \
    datapath/datasets/MUGE/valid_eval.json




#------------------------------------------------------------------------------
cd /home/zy/Downloads/data_process/Chinese-CLIP
  conda activate ultralytics

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





HARD_EVAL_SOURCE_SPLIT=train \
  bash run_scripts/evaluate_police_retrieval.sh \
    datapath/experiments/实验名/checkpoints/epoch4.pt \
    实验名_epoch4_trainhard




#最优
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  GPUS_PER_NODE=4 \
  BATCH_SIZE=8 \
  VALID_BATCH_SIZE=8 \
  HARD_NEGATIVE_WEIGHT=0.025 \
  EXP_NAME=police_retrieval_cleaned_hardneg0025_vit-b-16_g4_globalbs32 \
  bash run_scripts/custom_muge_finetune_1gpu.sh

  和：

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  GPUS_PER_NODE=4 \
  BATCH_SIZE=8 \
  VALID_BATCH_SIZE=8 \
  HARD_NEGATIVE_WEIGHT=0.1 \
  EXP_NAME=police_retrieval_cleaned_hardneg01_vit-b-16_g4_globalbs32 \
  bash run_scripts/custom_muge_finetune_1gpu.sh



#跑完评估后，把生成的 valid_eval_*.json 和 valid_bucket_eval_*.csv 拷回来，我再帮你定最终用哪个checkpoint。