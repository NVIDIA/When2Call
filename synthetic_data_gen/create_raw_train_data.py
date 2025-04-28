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
import uuid

from tqdm import tqdm

from utils import (
    OpenAIAPIPrompter,
    create_toolcall_answer,
    create_refusal_answer,
    create_rfi_answer,
    create_modified_tool_rfi_answer,
    create_direct_answer,
    rewrite_question_to_exclude_param,
    read_apigen_source_questions_and_answers,
)

random.seed(1234)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apigen_data_path",
        type=str,
        required=True,
        help="Path to directory containing APIGen data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory where data will be saved.",
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

    openai_api_prompter = OpenAIAPIPrompter(
        base_url=args.openai_api_base_url,
        api_key=os.environ.get("OPENAI_API_KEY"),
        model=args.openai_api_model,
        temperature=0.6,
        top_p=0.95,
    )

    train_data = []

    source_questions, source_answers = read_apigen_source_questions_and_answers(args.apigen_data_path)
    for src_idx, src_q in enumerate(tqdm(source_questions)):

        # Question with Refusal as correct answer
        refusal_question = {
            "uuid": str(uuid.uuid4()),
            "source": "APIGen",
            "source_id": src_q["id"],
            "question": src_q["query"],
            "orig_question": src_q["query"],
            "correct_answer": "cannot_answer",
            "answers": {
                "direct": create_direct_answer(src_q["query"], openai_api_prompter),
                "tool_call": {"name": source_answers[src_idx]["name"], "arguments": source_answers[src_idx]["arguments"]},
                "request_for_info": create_rfi_answer(src_q["query"], src_q["correct_tool"], openai_api_prompter),
                "cannot_answer": create_refusal_answer(src_q["query"], openai_api_prompter),
            },
            "target_tool": None,
            "tools": [json.dumps(func) for func in src_q["tools"] if func["name"] != src_q["correct_tool"]["name"]],
            "orig_tools": src_q["tools"],
            "held_out_param": None,
        }
        train_data.append(refusal_question)

        # Question with RFI as correct answer
        rewritten_question, required_param_to_remove = rewrite_question_to_exclude_param(src_q["query"], src_q["correct_tool"], openai_api_prompter)
        if rewritten_question and required_param_to_remove:
            rfi_question = {
                "uuid": str(uuid.uuid4()),
                "source": "APIGen",
                "source_id": src_q["id"],
                "question": rewritten_question,
                "orig_question": src_q["query"],
                "correct_answer": "request_for_info",
                "answers": {
                    "direct": create_direct_answer(src_q["query"], openai_api_prompter),
                    "tool_call": {"name": source_answers[src_idx]["name"], "arguments": source_answers[src_idx]["arguments"]},
                    "request_for_info": create_modified_tool_rfi_answer(rewritten_question, src_q["correct_tool"], required_param_to_remove, openai_api_prompter),
                    "cannot_answer": create_refusal_answer(src_q["query"], openai_api_prompter),
                },
                "target_tool": src_q["correct_tool"],
                "tools": src_q["tools"],
                "orig_tools": src_q["tools"],
                "held_out_param": required_param_to_remove,
            }
            train_data.append(rfi_question)

        # Question with Tool Call as correct answer
        toolcall_question = {
            "uuid": str(uuid.uuid4()),
            "source": "APIGen",
            "source_id": src_q["id"],
            "question": rewritten_question,
            "orig_question": src_q["query"],
            "correct_answer": "tool_call",
            "answers": {
                "direct": create_direct_answer(src_q["query"], openai_api_prompter),
                "tool_call": {"name": source_answers[src_idx]["name"], "arguments": source_answers[src_idx]["arguments"]},
                "request_for_info": create_rfi_answer(src_q["query"], src_q["correct_tool"], openai_api_prompter),
                "cannot_answer": create_refusal_answer(src_q["query"], openai_api_prompter),
            },
           "target_tool": src_q["correct_tool"],
            "tools": src_q["tools"],
            "orig_tools": src_q["tools"],
            "held_out_param": None,
        }
        train_data.append(toolcall_question)

    # Save raw train data
    with open(os.path.join(args.output_dir, "when2call_train_raw.jsonl"), "w") as f:
        for question in train_data:
            f.write(json.dumps(question)+"\n")



if __name__ == "__main__":
    main()
