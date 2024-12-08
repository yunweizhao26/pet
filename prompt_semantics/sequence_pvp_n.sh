#!/bin/bash
python3 src/encoder_decoder.py \
    --brand t5-3b \
    --save-dir runs/RTE_T5_3B/ \
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
    --seed 0,32,42 \
