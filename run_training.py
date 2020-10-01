# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to train and evaluate either a regular supervised model or a PET model on
one of the supported tasks and datasets.
"""

import os
import statistics
from collections import defaultdict
import torch
import numpy as np
from transformers.data.metrics import simple_accuracy

from argparsing import parser
from tasks import PROCESSORS, load_examples
from utils import set_seed, eq_div, save_logits, LogitsList, InputExample
from wrapper import TransformerModelWrapper
import log

logger = log.get_logger('root')


def main():
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.output_mode = "classification"
    args.label_list = processor.get_labels()
    args.use_logits = args.logits_file is not None

    wrapper = None

    logger.info("Training/evaluation parameters: {}".format(args))
    results = defaultdict(list)
    test_logits = []

    train_examples_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
    test_examples_per_label = eq_div(args.test_examples, len(args.label_list)) if args.test_examples != -1 else -1

    train_data = load_examples(args.task_name, args.data_dir, train_examples_per_label, evaluate=False)
    eval_data = load_examples(args.task_name, args.data_dir, test_examples_per_label, evaluate=True)

    if args.lm_training or args.save_train_logits or args.use_logits:
        all_train_data = load_examples(args.task_name, args.data_dir, args.lm_train_examples_per_label, evaluate=False)
    else:
        all_train_data = None

    if args.use_logits:
        logits = LogitsList.load(args.logits_file).logits
        assert len(logits) == len(all_train_data)
        logger.info("Got {} logits from file {}".format(len(logits), args.logits_file))
        for example, example_logits in zip(all_train_data, logits):
            example.logits = example_logits

    for pattern_id in args.pattern_ids:
        args.pattern_id = pattern_id
        for iteration in range(args.repetitions):

            results_dict = {}

            output_dir = "{}/p{}-i{}".format(args.output_dir, args.pattern_id, iteration)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if args.do_train:
                wrapper = TransformerModelWrapper(args)
                wrapper.model.to(device)

                results_dict['train_set_before_training'] = wrapper.eval(train_data, device, **vars(args))['acc']

                pattern_iter_train_data = []
                pattern_iter_train_data.extend(train_data)

                if args.additional_data_dir:
                    p = os.path.join(args.additional_data_dir, 'p{}-i{}-train.txt'.format(args.pattern_id, iteration))
                    additional_data = InputExample.load_examples(p)
                    for example in additional_data:
                        example.logits = None
                    pattern_iter_train_data.extend(additional_data)
                    logger.info("Loaded {} additional examples from {}, total training size is now {}".format(
                        len(additional_data), p, len(pattern_iter_train_data)
                    ))

                logger.info("Starting training...")

                global_step, tr_loss = wrapper.train(
                    pattern_iter_train_data, device,
                    helper_train_data=all_train_data if args.lm_training or args.use_logits else None,
                    tmp_dir=output_dir, **vars(args))

                logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
                logger.info("Training complete")

                results_dict['train_set_after_training'] = wrapper.eval(train_data, device, **vars(args))['acc']

                with open(os.path.join(output_dir, 'results.txt'), 'w') as fh:
                    fh.write(str(results_dict))

                logger.info("Saving trained model at {}...".format(output_dir))
                wrapper.save(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving complete")

                if args.save_train_logits:
                    logits, _ = wrapper.eval(all_train_data, device, output_logits=True, **vars(args))[0]
                    save_logits(os.path.join(output_dir, 'logits.txt'), logits)

                if not args.do_eval:
                    wrapper.model = None
                    wrapper = None
                    torch.cuda.empty_cache()

            # Evaluation
            if args.do_eval:
                logger.info("Starting evaluation...")
                if not wrapper:
                    wrapper = TransformerModelWrapper.from_pretrained(output_dir)
                    wrapper.model.to(device)

                logits, result = wrapper.eval(eval_data, device, output_logits=True, **vars(args))
                test_logits.append(logits)
                save_logits(os.path.join(output_dir, 'test_logits.txt'), logits)
                logger.info("--- RESULT (pattern_id={}, iteration={}) ---".format(pattern_id, iteration))
                logger.info(result)

                results_dict['test_set_after_training'] = result['acc']
                with open(os.path.join(output_dir, 'results.txt'), 'w') as fh:
                    fh.write(str(results_dict))

                for key, value in result.items():
                    results['{}-p{}'.format(key, args.pattern_id)].append(value)

                wrapper.model = None
                torch.cuda.empty_cache()

    logger.info("=== OVERALL RESULTS ===")

    with open(os.path.join(args.output_dir, 'result_test.txt'), 'w') as fh:
        for key, values in results.items():
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0
            result_str = "{}: {} +- {}".format(key, mean, stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')

        all_results = [result for pattern_results in results.values() for result in pattern_results]
        all_mean = statistics.mean(all_results)
        all_stdev = statistics.stdev(all_results)
        result_str = "acc-all-p: {} +- {}".format(all_mean, all_stdev)
        logger.info(result_str)
        fh.write(result_str + '\n')

        if args.ensembling is not None:
            labels = np.array([f.label for f in wrapper._convert_examples_to_features(eval_data, True)])
            if args.ensembling == "sum":
                total_logits = sum(test_logits)
                preds = np.argmax(total_logits, axis=1)
                acc = simple_accuracy(preds, labels)
            else:
                raise NotImplementedError
                # all_preds = [np.argmax(logits, axis=1) for logits in test_logits]
            ensembled_result_str = "acc-ens-p: {}".format(acc)
            logger.info(ensembled_result_str)
            fh.write(ensembled_result_str + '\n')


if __name__ == "__main__":
    main()
