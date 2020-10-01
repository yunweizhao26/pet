import argparse

from tasks import PROCESSORS
from wrapper import WRAPPER_TYPES

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--train_examples", required=True, type=int,
                    help="The total number of train examples to use, where -1 equals all examples.")
parser.add_argument("--wrapper_type", required=True, choices=WRAPPER_TYPES,
                    help="The wrapper type - either sequence_classifier (corresponding to"
                         "regular supervised training) or mlm (corresponding to PET training)")
parser.add_argument("--data_dir", default=None, type=str, required=True,
                    help="The input data dir. Should contain the data files for the task.")
parser.add_argument("--model_type", default=None, type=str, required=True,
                    help="The model type (currently supported are bert and roberta)")
parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                    help="Path to pre-trained model or shortcut name")
parser.add_argument("--task_name", default=None, type=str, required=True,
                    help="The name of the task to train selected in the list: " + ", ".join(PROCESSORS.keys()))
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")

# Optional parameters
parser.add_argument("--test_examples", default=-1, type=int,
                    help="The total number of test examples to use, where -1 equals all examples.")
parser.add_argument("--lm_train_examples_per_label", default=10000, type=int,
                    help="The total number of training examples for auxiliary language modeling, "
                         "where -1 equals all examples")
parser.add_argument("--pattern_ids", default=[0], type=int, nargs='+',
                    help="The ids of the PVPs to be used (only for PET training)")
parser.add_argument("--repetitions", default=3, type=int,
                    help="The number of times to repeat training and testing with different seeds.")
parser.add_argument("--ensembling", default=None, type=str, choices=['sum', 'vote'],
                    help="Whether to calculate the ensembled performance of all trained models")
parser.add_argument("--lm_training", action='store_true',
                    help="Whether to use language modeling as auxiliary task (only for PET training)")
parser.add_argument("--save_train_logits", action='store_true',
                    help="Whether to save logits on the lm_train_examples in a separate file. This takes some "
                         "additional time but is required for combining PVPs  (only for PET training)")
parser.add_argument("--additional_data_dir", default=None, type=str,
                    help="Path to a directory containing additional automatically labeled training examples (only "
                         "for iPET)")
parser.add_argument("--per_gpu_helper_batch_size", default=4, type=int,
                    help="Batch size for the auxiliary task (only for PET training)")
parser.add_argument("--alpha", default=0.9999, type=float,
                    help="Weighting term for the auxiliary task (only for PET training)")
parser.add_argument("--temperature", default=1, type=float,
                    help="Temperature used for combining PVPs (only for PET training)")
parser.add_argument("--verbalizer_file", default=None,
                    help="The path to a file to override default verbalizers (only for PET training)")
parser.add_argument("--logits_file", type=str,
                    help="The logits file for combining multiple PVPs, which can be created using the"
                         "merge_logits.py script")
parser.add_argument("--cache_dir", default="", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--do_lower_case", action='store_true',
                    help="Whether to perform lower casing")
parser.add_argument("--max_seq_length", default=256, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                         "than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--learning_rate", default=1e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=3.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument('--logging_steps', type=int, default=50,
                    help="Log every X updates steps.")
parser.add_argument("--no_cuda", action='store_true',
                    help="Avoid using CUDA when available")
parser.add_argument('--overwrite_output_dir', action='store_true',
                    help="Overwrite the content of the output directory")
parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument('--do_train', action='store_true',
                    help="Whether to perform training")
parser.add_argument('--do_eval', action='store_true',
                    help="Whether to perform evaluation")
