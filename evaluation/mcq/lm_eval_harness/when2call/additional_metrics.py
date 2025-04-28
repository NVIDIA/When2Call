# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
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

import json
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
import pandas as pd


def calculate_hallucination_rate(samples_path: str, print_rate=True) -> float:
    """
    In cases where the correct answer is "cannot answer" and no tools are provided,
    any tool call is a hallucination.
    This method calculates the percentage of times the model chose "tool call" as the predicted answer in these cases.
    Lower is better for hallucination rate.
    """
    results = []
    with open(samples_path) as samples_file:
        for line in samples_file:
            results.append(json.loads(line))

    filtered_results = [result for result in results
                        if result['doc']['correct_answer'] == "cannot_answer"
                        and len(result['doc']['tools']) == 0]
    if len(filtered_results) > 0:
        # The tool call is always the answer at index 1
        # Macro f1 contains (gold_index, predicted_index)
        # So the model predicted a tool call if result['macro_f1'][1] == 1
        hallucinated = [result for result in filtered_results if
                        result['macro_f1'][1] == 1]
        rate = len(hallucinated) / len(filtered_results)
        if print_rate:
            print(f'Hallucination rate: {rate:.2}')

        return rate
    else:
        return 0.0


def calculate_confusion_matrix(samples_path: str, print_matrix=True) -> dict:
    gold = []
    predicted = []
    label_names = []

    with open(samples_path) as samples_file:
        for i, line in enumerate(samples_file):
            sample_dict = json.loads(line)
            if i == 0:
                label_names = list(sample_dict['doc']['answers'].keys())

            gold.append(sample_dict['macro_f1'][0])
            predicted.append(sample_dict['macro_f1'][1])

    conf_matrix = confusion_matrix(gold, predicted)
    labelled_confusion_matrix = pd.DataFrame(conf_matrix,
                                             index=['true:{:}'.format(x) for x in label_names],
                                             columns=['pred:{:}'.format(x) for x in label_names]
                                             )
    if print_matrix:
        print(labelled_confusion_matrix.to_string())

    return json.loads(labelled_confusion_matrix.to_json(orient='index'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--samples_path', type=str, required=True,
                        help='Path to samples logged using the --log_samples flag (JSONL format)')

    args = parser.parse_args()

    conf_mat = calculate_confusion_matrix(args.samples_path)
    hall_rate = calculate_hallucination_rate(args.samples_path)

    print({
        'confusion_matrix': conf_mat,
        'hallucination_rate': hall_rate
    })
