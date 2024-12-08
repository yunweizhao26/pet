Comparisons and evaluations of few-shot fine-tuning with in-context learning with 27 handcrafted prompts. We choose tasks and datasets as follows:
- entailment: MNLI, RTE, CB
- multiple choice QA: BoolQ, MultiRC
- commonsense reasoning: WSC, COPA, WiC

prompts-across-models branch includes codebase for getting experiment results with few-shot finetuning and in-context learning. We did more than 5000 experiments for evaluation.

Test branch includes codebase for analysis. We did comparison the two methods using four correlation metrics and built correlation table for each experiment setup. Summaries of the results can be found in experiments/summary folder.