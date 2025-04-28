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
import random

from openai import OpenAI

from prompts import (
    SYNTH_REFUSAL_ANSWER_GEN_PROMPT,
    SYNTH_RFI_ANSWER_GEN_PROMPT,
    SYNTH_MODIFIED_TOOL_RFI_ANSWER_GEN_PROMPT,
    SYNTH_DIRECT_ANSWER_GEN_PROMPT,
    SYNTH_RFI_QUESTION_GEN_PROMPT,
)


class OpenAIAPIPrompter:
    def __init__(self, base_url: str,  api_key: str, model: str, temperature=0.6, top_p=0.95):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def single_prompt(self, prompt: str, max_tokens=256) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens
        )

        if completion.choices[0].finish_reason != 'stop':
            if completion.choices[0].finish_reason == "length":
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=max_tokens*2
                )
            else:
                raise ValueError(f'Expected model response to finish with stop token but model finish reason was: '
                                f'{completion.choices[0].finish_reason}.\n'
                                f'Prompt: {prompt}')

        response = completion.choices[0].message.content

        return response


def create_toolcall_answer(user_question, possible_answer):
    gt = possible_answer["ground_truth"][0]
    function_name = list(gt.keys())[0]
    toolcall = {"name": function_name, "arguments": {}}
    for param in gt[function_name]:
        sampled_value = gt[function_name][param][0]
        if sampled_value:
            toolcall["arguments"][param] = sampled_value

    return toolcall


def create_refusal_answer(user_question, openai_api_prompter):
    prompt = SYNTH_REFUSAL_ANSWER_GEN_PROMPT.format(user_question)
    response = openai_api_prompter.single_prompt(prompt)

    return response


def create_rfi_answer(user_question, tool, openai_api_prompter):
    prompt = SYNTH_RFI_ANSWER_GEN_PROMPT.format(tool, user_question)
    response = openai_api_prompter.single_prompt(prompt)

    return response


def create_modified_tool_rfi_answer(rewritten_question, tool, removed_param, openai_api_prompter):
    prompt = SYNTH_MODIFIED_TOOL_RFI_ANSWER_GEN_PROMPT.format(
        tool,
        removed_param,
        removed_param,
        rewritten_question,
    )
    response = openai_api_prompter.single_prompt(prompt)

    return response


def create_direct_answer(user_question, openai_api_prompter):
    prompt = SYNTH_DIRECT_ANSWER_GEN_PROMPT.format(user_question)
    response = openai_api_prompter.single_prompt(prompt)

    return response


def rewrite_question_to_exclude_param(user_question, tool, openai_api_prompter):
    orig_question = user_question
    required_params = tool["parameters"]["required"]

    if not required_params:
        return None, None

    required_param_to_remove = random.sample(required_params, 1)[0]
    prompt = SYNTH_RFI_QUESTION_GEN_PROMPT.format(
        required_param_to_remove,
        user_question,
        required_param_to_remove,
    )

    rewritten_question = openai_api_prompter.single_prompt(prompt)

    return rewritten_question, required_param_to_remove


def read_bfcl_source_questions(questions_source_filepath):
    source_questions = []
    with open(questions_source_filepath, "r") as f:
        for line in f:
            item = json.loads(line)
            source_questions.append(item)

    return source_questions


def read_bfcl_source_answers(answers_source_filepath):
    source_answers = []
    with open(answers_source_filepath, "r") as f:
        for line in f:
            item = json.loads(line)
            source_answers.append(item)

    return source_answers


def read_apigen_source_questions_and_answers(apigen_source_filepath):
    source_questions, source_answers = [], []
    with open(apigen_source_filepath, "r") as f:
        data = json.load(f)
        for item in data:
            item["answers"] = json.loads(item["answers"])
            item["tools"] = json.loads(item["tools"])

            # exclude parallel tool calls
            if len(item["answers"]) > 1:
                continue

            for idx, tool in enumerate(item["tools"]):
                properties = item["tools"][idx]["parameters"]
                item["tools"][idx]["parameters"] = {}
                item["tools"][idx]["parameters"]["type"] = "dict"
                item["tools"][idx]["parameters"]["properties"] = properties

                required_params = []
                for param in tool["parameters"]["properties"]:
                    if "optional" in tool["parameters"]["properties"][param]["type"]:
                        continue
                    required_params.append(param)
                item["tools"][idx]["parameters"]["required"] = required_params

            item["correct_tool"] = [tool for tool in item["tools"] if tool["name"] == item["answers"][0]["name"]][0]
            required_params = []
            for param in item["correct_tool"]["parameters"]["properties"]:
                if "optional" in item["correct_tool"]["parameters"]["properties"][param]["type"]:
                    continue
                required_params.append(param)
            item["correct_tool"]["parameters"]["required"] = required_params

            source_questions.append(item)
            source_answers.append(item["answers"][0])

    return source_questions, source_answers
