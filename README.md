# How many data points is a prompt worth?

This is companion code for our NAACL 2021 paper, "How many data points is a prompt worth?". Check out our [interactive blog post](https://huggingface.co/blog/how_many_data_points/) and [paper](https://arxiv.org/abs/2103.08493) for more details.

You should place your MNLI/SuperGLUE data in `data` in folders named after each task. For SuperGLUE data, run `python superglue_data_splitting` first in order to have a dev/test split.

You may run a _prompted_ experiment with `.scripts/sequential_pvps.sh` and a _head_ experiment with `.scripts/sequential_supervised.sh`. Modify those scripts to change the SuperGLUE task you wish to train on.

Originally forked from https://github.com/timoschick/pet.


srun -p gpu-rtx6k -A h2lab --time 1:00:00 -n 1 --mem=32G --gpus=1 --pty /bin/bash
