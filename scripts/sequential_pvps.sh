python3 cli.py \
  --method pet \
  --pattern_ids 0 \
  --data_dir data/ \
  --model_type roberta \
  --model_name_or_path roberta-large \
  --task_name mnli \
  --output_dir experiments/[TASK_NAME]/roberta/supervised \
  --do_train \
  --do_eval \
  --pet_per_gpu_eval_batch_size 4 \
  --pet_per_gpu_train_batch_size 4 \
  --pet_gradient_accumulation_steps 4 \
  --pet_num_train_epochs 10 \
  --pet_min_steps 250 \
  --pet_max_seq_length 256 \
  --pet_repetitions 1 \
  --sc_per_gpu_train_batch_size 4 \
  --sc_per_gpu_unlabeled_batch_size 16 \
  --sc_gradient_accumulation_steps 4 \
  --sc_num_train_epochs 10 \
  --sc_min_steps 250 \
  --sc_max_seq_length 256 \
  --sc_repetitions 1 \
  --train_examples 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 \
  --warmup_steps 50 \
  --logging_steps 50 \
  --overwrite_output_dir \
  --no_distillation
