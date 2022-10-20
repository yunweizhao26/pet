#!/bin/bash
for model in roberta-large roberta-base
do
  for id in 3 4 5 6 7 8 9
  do
    python3 cli.py \
      --method pet \
      --pattern_ids $id \
      --data_dir split_data/ \
      --model_type roberta \
      --model_name_or_path $model \
      --task_name rte \
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
      --no_distillation
  done
done