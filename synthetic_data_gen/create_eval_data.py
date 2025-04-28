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
    read_bfcl_source_questions,
    read_bfcl_source_answers,
)

random.seed(1234)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bfcl_data_dir",
        type=str,
        required=True,
        help="Path to directory containing BFCL data.",
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

    eval_data = []

    bfcl_categories = ["BFCL_v2_live_multiple", "BFCL_v2_live_simple"]
    for bfcl_category in bfcl_categories[:1]:
        questions_source_filepath = os.path.join(args.bfcl_data_dir, f"{bfcl_category}.json")
        answers_source_filepath = os.path.join(args.bfcl_data_dir, f"possible_answer/{bfcl_category}.json")

        source_questions = read_bfcl_source_questions(questions_source_filepath)
        source_answers = read_bfcl_source_answers(answers_source_filepath)
        assert len(source_questions) == len(source_answers)

        for src_idx, src_q in enumerate(tqdm(source_questions[:1])):
            if not isinstance(src_q["function"], list):
                src_q["function"] = [src_q["function"]]

            target_tool = [tool for tool in src_q["function"] if tool["name"] in source_answers[src_idx]["ground_truth"][0]][0]

            # Question with Refusal as correct answer
            refusal_question = {
                "uuid": str(uuid.uuid4()),
                "source": bfcl_category,
                "source_id": src_q["id"],
                "question": src_q["question"][-1]["content"],
                "orig_question": src_q["question"][-1]['content'],
                "correct_answer": "cannot_answer",
                "answers": {
                    "direct": create_direct_answer(src_q["question"][-1]['content'], openai_api_prompter),
                    "tool_call": create_toolcall_answer(src_q["question"][-1]['content'], source_answers[src_idx]),
                    "request_for_info": create_rfi_answer(src_q["question"][-1]['content'], target_tool, openai_api_prompter),
                    "cannot_answer": create_refusal_answer(src_q["question"][-1]['content'], openai_api_prompter),
                },
                "target_tool": None,
                "tools": [json.dumps(func) for func in src_q["function"] if func["name"] != target_tool["name"]],
                "orig_tools": [json.dumps(func) for func in src_q["function"]],
                "held_out_param": None,
            }
            eval_data.append(refusal_question)

            # Question with RFI as correct answer
            if target_tool["parameters"]["required"]:
                rewritten_question, required_param_to_remove = rewrite_question_to_exclude_param(src_q["question"][-1]['content'], target_tool, openai_api_prompter)
                rfi_question = {
                    "uuid": str(uuid.uuid4()),
                    "source": bfcl_category,
                    "source_id": src_q["id"],
                    "question": rewritten_question,
                    "orig_question": src_q["question"][-1]['content'],
                    "correct_answer": "request_for_info",
                    "answers": {
                        "direct": create_direct_answer(src_q["question"][-1]['content'], openai_api_prompter),
                        "tool_call": create_toolcall_answer(src_q["question"][-1]['content'], source_answers[src_idx]),
                        "request_for_info": create_modified_tool_rfi_answer(rewritten_question, target_tool, required_param_to_remove, openai_api_prompter),
                        "cannot_answer": create_refusal_answer(src_q["question"][-1]['content'], openai_api_prompter),
                    },
                    "target_tool": target_tool,
                    "tools": [json.dumps(func) for func in src_q["function"]],
                    "orig_tools": [json.dumps(func) for func in src_q["function"]],
                    "held_out_param": required_param_to_remove,
                }
                eval_data.append(rfi_question)

            # Question with Tool Call as correct answer
            toolcall_question = {
                "uuid": str(uuid.uuid4()),
                "source": bfcl_category,
                "source_id": src_q["id"],
                "question": src_q["question"][-1]['content'],
                "orig_question": src_q["question"][-1]['content'],
                "correct_answer": "tool_call",
                "answers": {
                    "direct": create_direct_answer(src_q["question"][-1]['content'], openai_api_prompter),
                    "tool_call": create_toolcall_answer(src_q["question"][-1]['content'], source_answers[src_idx]),
                    "request_for_info": create_rfi_answer(src_q["question"][-1]['content'], target_tool, openai_api_prompter),
                    "cannot_answer": create_refusal_answer(src_q["question"][-1]['content'], openai_api_prompter),
                },
                "target_tool": json.dumps(target_tool),
                "tools": [json.dumps(func) for func in src_q["function"]],
                "orig_tools": [json.dumps(func) for func in src_q["function"]],
                "held_out_param": None,
            }

            eval_data.append(toolcall_question)

    random.shuffle(eval_data)

    # Save MCQ eval data
    with open(os.path.join(args.output_dir, "when2call_test_mcq.jsonl"), "w") as f:
        for question in eval_data:
            f.write(json.dumps(question)+"\n")

    # Save LLM-as-a-Judge eval data
    questions2category = {}
    for question in eval_data:
        if question["correct_answer"] not in questions2category:
            questions2category[question["correct_answer"]] = []
        questions2category[question["correct_answer"]].append(question)

    selected_questions = {}
    for category in questions2category:
        random.shuffle(questions2category[category])
        selected_questions[category] = questions2category[category][:100]

    with open(os.path.join(args.output_dir, "when2call_test_llm-as-a-judge.jsonl"), "w") as f:
        for category in selected_questions:
            for question in selected_questions[category]:
                f.write(json.dumps(question)+"\n")


if __name__ == "__main__":
    main()
