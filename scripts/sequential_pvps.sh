#!/bin/bash
for model in roberta-large roberta-base
do
  for id in {0..26}
  do
    python cli.py \
      --method pet \
      --pattern_ids $id \
      --data_dir split_data/ \
      --model_typet \
      --model_name_or_path $model \
      --task_name mnli \
      --output_dir experiments/[TASK_NAME]/$model/supervised \
      --do_train \
      --do_eval \
      --do_test \
      --pet_per_gpu_eval_batch_size 4 \
      --pet_per_gpu_train_batch_size 4 \
      --pet_gradient_accumulation_steps 4 \
      --pet_num_train_epochs 10 \
      --pet_min_steps 250 \
      --pet_max_steps 2000 \
      --pet_max_seq_length 256 \
      --pet_repetitions 1 \
      --sc_per_gpu_train_batch_size 4 \
      --sc_per_gpu_unlabeled_batch_size 16 \
      --sc_gradient_accumulation_steps 4 \
      --sc_num_train_epochs 10 \
      --sc_min_steps 250 \
      --sc_max_steps 2000 \
      --sc_max_seq_length 256 \
      --sc_repetitions 1 \
      --train_examples 32 \
      --warmup_steps 50 \
      --logging_steps 50 \
      --overwrite_output_dir \
      --seed 42 \
      --cache_dir ~/../../../gscratch/cse/yunwei/cache/ \
      --wrapper_type plm \
      --no_distillation
  done
done