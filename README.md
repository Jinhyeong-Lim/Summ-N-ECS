# Summ-N-ECS
Source code for EMNLP 2023 Workshop paper [Improving Multi-Stage Long Document Summarization with Enhanced Coarse Summarizer](https://aclanthology.org/2023.newsum-1.13/)

### Dependency
- We use python==3.10.8, pytorch==1.13.1, transformers==4.25.1 and evaluate==0.4.0

### Folder Structure

- configure: the running configures for each dataset, such as number of stages, beam width etc.
- dataset_loader: the python scripts to convert original dataset to the uniform format.
- models: model
  - data_segment: including source and target segmentation code;
- utils: utilities such as config parser & dataset reader etc.

### Download the Datasets and Models
- Download link for AMI & ICSI can be found at https://github.com/microsoft/HMNet
 
### Data generation
Preprocess ICSI summarization dataset

```bash
python icsi_preprocess.py
```

### Coarse-stage Fine-tuning
```bash
python icsi_ctr.py \
  --do_train \
  --do_eval \
  --do_predict \
  --report_to "none" \
  --model_name dia \
  --train_file  ./Summ-N-ECS/icsi/stage_1/train_rouge12L.csv \
  --validation_file  ./Summ-N-ECS/icsi/stage_1/val_rouge12L.csv \
  --test_file  ./Summ-N-ECS/icsi/stage_1/test_rouge12L.csv \
  --max_source_length 1024 \
  --output_dir ./Summ-N-ECS/icsi/rouge12L/model \
  --learning_rate 2e-05 \
  --warmup_ratio 0.1 \
  --gradient_accumulation_steps 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 16 \
  --max_target_length 300 \
  --val_max_target_length 300 \
  --num_train_epochs 30 \
  --save_strategy epoch \
  --lr_scheduler_type 'polynomial' \
  --max_grad_norm 0.1 \
  --evaluation_strategy epoch \
  --dataloader_num_workers 10 \
  --save_steps 500 \
  --num_beams 6 \
  --weight_decay 0.1 \
  --eval_steps 500 \
  --logging_steps 500 \
  --save_total_limit 3 \
  --load_best_model_at_end \
  --label_smoothing_factor 0.1 \
  --overwrite_cache \
  --fp16 \
  --seed 42 \
  --predict_with_generate 
```

### Coarse-stage Inference
```bash
CUDA_VISIBLE_DEVICES=1 python icsi_inf.py \
  --do_predict \
  --report_to "none" \
  --model_name “best model path” \
  --train_file  ./Summ-N-ECS/icsi/stage_1/train_rouge12L.csv \
  --validation_file  ./Summ-N-ECS/icsi/stage_1/val_rouge12L.csv \
  --test_file  ./Summ-N-ECS/icsi/stage_1/test_rouge12L.csv \
  --max_source_length 1024 \
  --output_dir ./Summ-N-ECS/icsi/rouge12L/model \
  --learning_rate 2e-05 \
  --warmup_ratio 0.1 \
   --gradient_accumulation_steps 1 \
  --metric_for_best_model "eval_loss" \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 16 \
  --max_target_length 300 \
  --val_max_target_length 300 \
  --num_train_epochs 30 \
  --save_strategy epoch \
  --lr_scheduler_type 'polynomial' \
  --max_grad_norm 0.1 \
  --evaluation_strategy epoch \
  --dataloader_num_workers 10 \
  --num_beams 6 \
  --weight_decay 0.1 \
  --eval_steps 500 \
  --logging_steps 500 \
  --save_total_limit 3 \
  --load_best_model_at_end \
  --label_smoothing_factor 0.1 \
  --overwrite_cache \
  --fp16 \
  --seed 42 \
  --predict_with_generate 
```


### Fine-grained stage Fine-tuning
```bash
CUDA_VISIBLE_DEVICES=1 python icsi_fine.py \
  --do_train \
  --do_eval \
  --do_predict \
  --report_to "none" \
  --model_name cnn \
  --train_file  ./Summ-N-ECS/icsi/rouge12L/coarse_summary/train.csv \
  --validation_file  ./Summ-N-ECS/icsi/rouge12L/coarse_summary/val.csv \
  --test_file  ./Summ-N-ECS/icsi/rouge12L/coarse_summary/test.csv \
  --max_source_length 1024 \
  --output_dir ./Summ-N-ECS/icsi/rouge12L/fine_model \
  --learning_rate 3e-05 \
  --warmup_steps 500 \
   --gradient_accumulation_steps 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 16 \
  --max_target_length 900 \
  --val_max_target_length 900 \
  --num_train_epochs 30 \
  --save_strategy epoch \
  --lr_scheduler_type 'polynomial' \
  --max_grad_norm 0.1 \
  --evaluation_strategy epoch \
  --dataloader_num_workers 10 \
  --num_beams 10 \
  --weight_decay 0.1 \
  --save_total_limit 3 \
  --load_best_model_at_end \
  --label_smoothing_factor 0.1 \
  --overwrite_cache \
  --fp16 \
  --seed 42 \
  --predict_with_generate 
```

