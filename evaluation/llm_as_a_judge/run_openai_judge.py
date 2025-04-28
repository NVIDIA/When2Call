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


JUDGE_PROMPT = """You are an expert at classifying responses from AI models.

Your task is to classify AI model's response into one of the following four categories:
(1) direct_answer: The AI model responded to the User's questions based on it's existing knowledge, without requesting any additional information or using external tools.
(2) tool_call: The AI model decided to use a tool from the provided one's to help answer the question.
(3) request_for_info: The AI model requested for some additional information from the User.
(4) cannot_answer: The AI model refused to answer the User's questions by acknowledging the lack of required capabilities.

*You should not judge whether the AI model's response is accurate or not. Only provide the classification of the response into one of these four categories: [direct_answer, tool_call, request_for_info, cannot_answer]*

- The tools available to the AI model are given in <AVAILABLE_TOOLS> </AVAILABLE_TOOLS>
- The User's question is provided in <USER_QUESTION> </USER_QUESTION>
- The AI model's response is provided in <AI_MODEL_RESPONSE> </AI_MODEL_RESPONSE> which may or may not invlove a tool call

<AVAILABLE_TOOLS>
{}
</AVAILABLE_TOOLS>

<USER_QUESTION>
{}
</USER_QUESTION>

<AI_MODEL_RESPONSE>
{}
</AI_MODEL_RESPONSE>

Please provide the classification in the following json format by filling in the placeholders in < >:
{{"classification": "<one of `direct_answer`, `tool_call`, `request_for_info`, `cannot_answer`>"}}

Respond only in the prescribed json format with the placeholders filled in."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--responses_path",
        type=str,
        required=True,
        help="Path to model responses on When2Call eval data to be judged.",
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
        "--judge_model",
        type=str,
        required=True,
        help="Model name to be used for synthetic data generation.",
    )

    args = parser.parse_args()

    client = OpenAI(base_url=args.openai_api_base_url, api_key=os.environ.get("OPENAI_API_KEY"))

    responses = []
    with open(args.responses_path, "r") as f:
        for line in f:
            item = json.loads(line)
            if item["target_tool"] is not None:
                item["target_tool"] = json.loads(item["target_tool"])
            item["tools"] = [json.loads(x) for x in item["tools"]]
            responses.append(item)

    existing = 0
    if os.path.exists(args.results_path):
        with open(args.results_path, "r") as f:
            for line in f:
                existing += 1

    for idx, response in enumerate(tqdm(responses)):

        if idx < existing:
            continue

        judge_prompt = JUDGE_PROMPT.format(
            response["tools"],
            response["question"],
            response["model_response"],
        )

        messages = [{"role": "user", "content": judge_prompt}]
        judge_response = client.chat.completions.create(
            model=args.judge_model,
            messages=messages,
        ).choices[0].message.content
        judge_response = judge_response.strip()

        try:
            judge_response = json.loads(judge_response)
        except:
            print(judge_response)
            messages.append({"role": "assistant", "content": judge_response})
            messages.append({"role": "user", "content": "Please re-write your response to be shorter and make sure it's a valid json in the prescribed format."})
            judge_response = client.chat.completions.create(
                model=args.judge_model,
                messages=messages,
            ).choices[0].message.content
            judge_response = judge_response.strip()
            judge_response = json.loads(judge_response)

        with open(args.results_path, "a") as f:
            item_to_dump = {
                "judge_prompt": judge_prompt,
                "judge_response": judge_response,
                "model_response": response["model_response"],
                "question": response,
            }
            f.write(json.dumps(item_to_dump)+"\n")


if __name__ == "__main__":
    main()
