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

import argparse
import json
import os
import random


random.seed(1234)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_train_data_path",
        type=str,
        required=True,
        help="Path to raw When2Call train data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory where data will be saved.",
    )

    args = parser.parse_args()

    formatted_data = []

    input_data = []
    with open(args.raw_train_data_path, "r") as f:
        for line in f:
            input_data.append(json.loads(line))

    random.shuffle(input_data)
    for item in input_data[:1000]:

        formatted_item = {
            "tools": item['tools'],
            "messages": [{"role": "user", "content": item['question']}],
            "chosen_reward": 1.0,
            "rejected_reward": 0.0,
        }

        if item["correct_answer"] == "tool_call":
            formatted_item["chosen_response"] = {"role": "assistant", "content": "<TOOLCALL>[" + f"{json.dumps(item['answers']['tool_call'])}" + "]</TOOLCALL>"}
    
            p = random.random()
            if p < (1/3):
                formatted_item["rejected_response"] = {"role": "assistant", "content": item['answers']['direct']}
            elif p < (2/3):
                formatted_item["rejected_response"] = {"role": "assistant", "content": item['answers']['request_for_info']}
            else:
                formatted_item["rejected_response"] = {"role": "assistant", "content": item['answers']['cannot_answer']}
        
        elif item["correct_answer"] == "cannot_answer":
            formatted_item["chosen_response"] = {"role": "assistant", "content": item['answers']['cannot_answer']}

            p = random.random()
            if p < (1/3):
                formatted_item["rejected_response"] = {"role": "assistant", "content": item['answers']['direct']}
            elif p < (2/3):
                formatted_item["rejected_response"] = {"role": "assistant", "content": item['answers']['request_for_info']}
            else:
                formatted_item["rejected_response"] = {"role": "assistant", "content": "<TOOLCALL>[" + f"{json.dumps(item['answers']['tool_call'])}" + "]</TOOLCALL>"}

        elif item["correct_answer"] == "request_for_info":
            formatted_item["chosen_response"] = {"role": "assistant", "content": item['answers']['request_for_info']}

            p = random.random()
            if p < (1/3):
                formatted_item["rejected_response"] = {"role": "assistant", "content": item['answers']['direct']}
            elif p < (2/3):
                formatted_item["rejected_response"] = {"role": "assistant", "content": "<TOOLCALL>[" + f"{json.dumps(item['answers']['tool_call'])}" + "]</TOOLCALL>"}
            else:
                formatted_item["rejected_response"] = {"role": "assistant", "content": item['answers']['cannot_answer']}

        formatted_data.append(formatted_item)

    random.shuffle(formatted_data)

    with open(os.path.join(args.output_dir, "when2call_train_preference.jsonl"), "w") as f:
        for item in formatted_data:
            f.write(json.dumps(item)+"\n")


if __name__ == "__main__":
    main()
