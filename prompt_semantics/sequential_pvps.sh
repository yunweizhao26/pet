#!/bin/bash
PYTHONPATH=../../src USE_TF=0 /usr/bin/time -v deepspeed --num_gpus=7 src/encoder_decoder.py \
    --brand t5-11b \
    --save-dir runs/RTE_T5_11B/ \
    --dataset rte \
    --prompt-path data/binary_NLI_prompts.csv \
    --experiment-name 'sec4' \
    --num-shots 32 \
    --epochs 10 \
    --train-batch-size 4 \
    --eval-batch-size 4 \
    --grad-accumulation 4 \
    --learning-rate 1e-5 \
    --production \
    --deepspeed_config ds_config.json \
    --seed 0,32,42 \
