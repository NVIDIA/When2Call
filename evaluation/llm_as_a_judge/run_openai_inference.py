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
import requests
import time

from tqdm import tqdm

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True,
        help="Path to When2Call eval data for LLM-as-a-Judge eval.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Filepath to save where results.",
    )
    parser.add_argument(
        "--openai_api_base_url",
        type=str,
        required=True,
        help="Base URL for OpenAI API to be used for synthetic data generation.",
    )
    parser.add_argument(
        "--openai_api_model",
        type=str,
        required=True,
        help="Model name to be used for synthetic data generation.",
    )

    args = parser.parse_args()

    client = OpenAI(base_url=args.openai_api_base_url, api_key=os.environ.get("OPENAI_API_KEY"))

    existing_count = 0
    if os.path.exists(args.results_path):
        with open(args.results_path, "r") as f:
            for line in f:
                existing_count += 1

    with open(args.eval_data_path, "r") as fin:
        for idx, line in enumerate(tqdm(fin.readlines())):
            if idx < existing_count:
                continue

            test_item = json.loads(line)
            tools = [json.loads(t.replace("float", "string").replace("integer", "string").replace("dict", "object").replace("tuple", "object")) for t in test_item["tools"]]
            tools = [{"type": "function", "function": t} for t in tools]
            for t in tools:
                t["function"]["name"] = t["function"]["name"].replace(".", "_")
                t["function"]["parameters"]["type"] = "object"
                for param in t["function"]["parameters"]["properties"]:
                    t["function"]["parameters"]["properties"][param]["type"] = "string"

            messages = [{"role": "user", "content": test_item["question"]}]

            if tools:
                responses = client.chat.completions.create(
                    model=args.openai_api_model,
                    messages=messages,
                    tools=tools,
                )
            else:
                responses = client.chat.completions.create(
                    model=args.openai_api_model,
                    messages=messages,
                )

            if responses.choices[0].message.tool_calls is not None:
                output = json.dumps({"name": responses.choices[0].message.tool_calls[0].function.name, "arguments": json.loads(responses.choices[0].message.tool_calls[0].function.arguments)})
            else:
                output = responses.choices[0].message.content.strip()

            with open(args.results_path, "a") as fout:
                test_item["model_response"] = output
                fout.write(f"{json.dumps(test_item, ensure_ascii=False)}\n")


if __name__ == "__main__":
    main()
