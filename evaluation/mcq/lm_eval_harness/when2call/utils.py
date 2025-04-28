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
from typing import Tuple, List, Callable

from datasets import Dataset

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. 
You have access to the following tools described in <tool></tool> which you can use to answer the user's questions.
Only use a tool if it directly answers the user's question.
"""

TOOL_USE_INSTRUCTIONS = """To use a tool, return JSON in the following format:
{"name": "tool_name", "arguments": {"argument1": "value1", "argument2": "value2", ...}}
"""


def get_choices_and_index(item: dict, format_tool_call: Callable[[str], str] = None) -> Tuple[List[str], int]:
    # Get ordered list of answers
    answer_names = list(item['answers'].keys())
    choices = []
    for name in answer_names:
        if 'tool' in name and format_tool_call:
            choices.append(format_tool_call(item['answers'][name]))
        else:
            choices.append(item['answers'][name])

    # Use ordered list to get answer index
    correct_answer_index = answer_names.index(item['correct_answer'])
    target_index = correct_answer_index

    return choices, target_index


def default_format_tools(tools: List[str]) -> str:
    tool_strings = [f'<tool>{tool}</tool>'
                    for tool in tools]
    tool_string = "\n\n".join(tool_strings)
    return tool_string


def process_docs_default(dataset: Dataset) -> Dataset:
    def _make_mc_list(item: dict) -> dict:
        choices, target_index = get_choices_and_index(item)
        item['choices'] = choices
        item['target_index'] = target_index

        # Format tools
        tool_string = default_format_tools(item['tools'])

        # Compose prompt (note no space at the end; config has target_delimiter = " ")
        item['prompt'] = f'{DEFAULT_SYSTEM_PROMPT}\n{TOOL_USE_INSTRUCTIONS}\n\n{tool_string}\n\n{item["question"]}'
        return item

    return dataset.map(_make_mc_list)


def process_docs_llama3_2(dataset: Dataset) -> Dataset:
    def _format_tool_call(tool_call_answer: str) -> str:
        tool_call_answer = json.loads(tool_call_answer)
        formatted = "["
        formatted += f"{tool_call_answer['name']}("
        for arg in tool_call_answer["arguments"]:
            formatted += f"{arg}="
            if isinstance(tool_call_answer['arguments'][arg], str):
                formatted += f"\"{tool_call_answer['arguments'][arg]}\", "
            else:
                formatted += f"{tool_call_answer['arguments'][arg]}, "
        if formatted.endswith(", "):
            formatted = formatted[:-2]
        formatted += f")]"
        return formatted

    def _make_mc_list(item: dict) -> dict:
        choices, target_index = get_choices_and_index(item, format_tool_call=_format_tool_call)
        item['choices'] = choices
        item['target_index'] = target_index
        item['prompt'] = f"""<|start_header_id|>system<|end_header_id|>

You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function,also point it out. You should only return the function call in tools call sections.
If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.
{json.dumps([json.loads(t) for t in item['tools']])}<|eot_id|><|start_header_id|>user<|end_header_id|>

{item['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return item

    return dataset.map(_make_mc_list)


def process_docs_qwen2_5(dataset: Dataset) -> Dataset:
    def _format_tool_call(tool_call_answer: str) -> str:
        return tool_call_answer

    def _make_mc_list(item: dict) -> dict:
        choices, target_index = get_choices_and_index(item, format_tool_call=_format_tool_call)
        item['choices'] = choices
        item['target_index'] = target_index
        tools_str = ""
        for tool in item['tools']:
            tools_str += tool + "\n"
        tools_str = tools_str.strip()

        item['prompt'] = f"""<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_str}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
{item['question']}<|im_end|>
<|im_start|>assistant
"""
        return item

    return dataset.map(_make_mc_list)


def process_docs_nemotron(dataset: Dataset) -> Dataset:
    """
    While LM Evaluation Harness does support chat templates (see `model_guide.md`),
    this doesn't support adding tool specifications to the system prompt, as
    the system prompt is expected to be constant.
    This function builds the
    """

    def _format_tool_call(tool_call_answer: str) -> str:
        return f'<toolcall> {tool_call_answer} </toolcall>'

    def _make_mc_list(item: dict) -> dict:
        choices, target_index = get_choices_and_index(item, format_tool_call=_format_tool_call)
        item['choices'] = choices
        item['target_index'] = target_index

        # Format tools
        tool_string = default_format_tools(item['tools'])

        # Compose prompt with chat template
        item['prompt'] = (f'<extra_id_0>System\n{DEFAULT_SYSTEM_PROMPT}\n\n{tool_string}\n\n'
                          f'<extra_id_1>User\n{item["question"]}\n<extra_id_1>Assistant\n')

        return item

    return dataset.map(_make_mc_list)


def process_docs_xlam(dataset: Dataset) -> Dataset:
    """Adapted from https://huggingface.co/Salesforce/xLAM-7b-fc-r#basic-usage-with-huggingface
    We use their suggested task and format instruction as-is, even though it may not match expectations by the benchmark
    (e.g. returning an empty list if no tool call is needed, rather than returning text)
    """

    task_instruction = """
    You are an expert in composing functions. You are given a question and a set of possible functions. 
    Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
    If none of the functions can be used, point it out and refuse to answer. 
    If the given question lacks the parameters required by the function, also point it out.
    """.strip()

    format_instruction = """
    The output MUST strictly adhere to the following JSON format, and NO other text MUST be included.
    The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make tool_calls an empty list '[]'.
    ```
    {
        "tool_calls": [
        {"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
        ... (more tool calls as required)
        ]
    }
    ```
    """.strip()

    def _format_tool_call(tool_call_answer: str) -> str:
        return f"{{\n\t\"tool_calls\": [\n\t{tool_call_answer}\n\t]\n}}"

    def _build_prompt(tools: List[str], query: str) -> str:
        prompt = f"[BEGIN OF TASK INSTRUCTION]\n{task_instruction}\n[END OF TASK INSTRUCTION]\n\n"
        prompt += f"[BEGIN OF AVAILABLE TOOLS]\n{tools}\n[END OF AVAILABLE TOOLS]\n\n"
        prompt += f"[BEGIN OF FORMAT INSTRUCTION]\n{format_instruction}\n[END OF FORMAT INSTRUCTION]\n\n"
        prompt += f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
        return prompt

    def _make_mc_list(item: dict) -> dict:
        choices, target_index = get_choices_and_index(item, format_tool_call=_format_tool_call)
        item['choices'] = choices
        item['target_index'] = target_index

        # Compose prompt without chat template; use lm-evaluation-harness chat template support
        item['prompt'] = _build_prompt(item['tools'], item['question'])

        return item

    return dataset.map(_make_mc_list)


def process_docs_hermes(dataset: Dataset):
    """Adapted from https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B#prompt-format-for-function-calling"""

    def _format_tool_call(tool_call_answer: str) -> str:
        return f"<tool_call>{tool_call_answer}\n</tool_call>"

    def _build_prompt(tools: List[str], query: str) -> str:
        prompt = "<|im_start|>system\n"
        prompt += "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: "
        prompt += f"<tools> {' '.join(tools)} </tools> "
        prompt += "Use the following pydantic model json schema for each tool call you will make: {\"properties\": {\"arguments\": {\"title\": \"Arguments\", \"type\": \"object\"}, \"name\": {\"title\": \"Name\", \"type\": \"string\"}}, \"required\": [\"arguments\", \"name\"], \"title\": \"FunctionCall\", \"type\": \"object\"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n"
        prompt += "<tool_call>\n{\"arguments\": <args-dict>, \"name\": <function-name>}\n</tool_call><|im_end|>"
        prompt += "<|im_start|>user\n"
        prompt += query
        prompt += "<|im_end|>"
        return prompt

    def _make_mc_list(item: dict) -> dict:
        choices, target_index = get_choices_and_index(item, format_tool_call=_format_tool_call)
        item['choices'] = choices
        item['target_index'] = target_index

        # Compose prompt with chat template; do not use lm-evaluation-harness chat template support since we
        # can't include tools in the system prompt
        item['prompt'] = _build_prompt(item['tools'], item['question'])

        return item

    return dataset.map(_make_mc_list)


def process_docs_functionary(dataset: Dataset):
    """Adapted from https://github.com/MeetKai/functionary/blob/main/tests/prompt_test_v3-llama3.1.txt"""

    system_prompt_start = """<|start_header_id|>system<|end_header_id|>

Environment: ipython

Cutting Knowledge Date: December 2023


You have access to the following functions:
"""

    system_prompt_end = """
Think very carefully before calling functions.
If a you choose to call a function ONLY reply in the following format:
<{start_tag}={function_name}>{parameters}{end_tag}
where

start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`

Here is an example,
<function=example_function_name>{"example_name": "example_value"}</function>

Reminder:
- If looking for real time information use relevant functions before falling back to brave_search
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line

<|eot_id|>"""

    def _format_tool_call(tool_call_answer: str) -> str:
        tool_call_dict = json.loads(tool_call_answer)
        tool_name = tool_call_dict['name']
        arguments_dict = tool_call_dict['arguments'] if 'arguments' in tool_call_dict else tool_call_dict['parameters']
        tool_call = f"<function={tool_name}>{{"
        tool_call += ', '.join([f'"{argument_name}": "{json.dumps(argument_value)}"'
                                for argument_name, argument_value in arguments_dict.items()])
        tool_call += "}</function>"
        return tool_call

    def _format_tool(tool_specification: str) -> str:
        tool_spec_dict = json.loads(tool_specification)
        tool_name = tool_spec_dict['name']
        tool_description = tool_spec_dict['description']

        tool_string = f"Use the function '{tool_name}' to '{tool_description}'\n"
        tool_string += tool_specification
        tool_string += '\n\n'

        return tool_string

    def _build_prompt(tools: List[str], query: str) -> str:
        prompt = system_prompt_start
        for tool in tools:
            prompt += _format_tool(tool)
        prompt += system_prompt_end
        prompt += "<|start_header_id|>user<|end_header_id|>\n\n"
        prompt += query
        prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    def _make_mc_list(item: dict) -> dict:
        choices, target_index = get_choices_and_index(item, format_tool_call=_format_tool_call)
        item['choices'] = choices
        item['target_index'] = target_index

        # Compose prompt with chat template; do not use lm-evaluation-harness chat template support since we
        # can't include tools in the system prompt
        item['prompt'] = _build_prompt(item['tools'], item['question'])

        return item

    return dataset.map(_make_mc_list)
