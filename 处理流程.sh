#cvat coco 数据集 ->  muge 数据集

python cvat2muge.py --input-dir ""  --output-dir "cvat_datasets" 


#cvat datasets 过滤长文本
python filter_cvat_datasets.py --dataset-dir "cvat_datasets"  --out-dir "cvat_no_long_datasets"  --max-length 55


#将搜集的pos neg 数据集里的pos数据集 生成train 的jsonl 和 tsv  text-start-id 和 image-start-id 需要根据cvat_no_long_datasets 里最后一个id 来设置，保证不重复
python build_muge_all_pos.py --root "/home/zy/Download/pos_neg_datasets" \
                            --out-dir "pos_datasets" \
                            --text-start-id xxxx \
                            --image-start-id xxxx \


#将pos dataset数据集 合并到cvat_no_long_datasets 里
python merge_pos_to_cavt.py --cvat-dir "cvat_no_long_datasets" \
                            --pos-dir "pos_datasets" \
                            --out-dir "cvat_merge_pos_datasets" \


#将cvat_merge_pos_datasets 里的数据集 进行扩展
python expand_train_texts.py --input "cvat_merge_pos_datasets" \
                            --output "cvat_merge_pos_datasets_expanded" \
                            --max-new-per-item 4 \
                            
