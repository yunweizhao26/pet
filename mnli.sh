python3 run_training.py \
--wrapper_type sequence_classifier \
--train_examples 100 \
--data_dir /home/teven/pet/data/MNLI/ \
--model_type roberta \
--model_name_or_path roberta-large \
--task_name mnli \
--output_dir /home/teven/pet/experiments/MNLI/ \
--do_train \
--do_eval 
