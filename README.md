# When2Call

When (not) to Call Tools

Leveraging external tools is a key feature for modern Language Models (LMs) to expand their capabilities and integrate them into existing systems. Various tool calling benchmarks have been developed to assess LMs' tool-calling capabilities. However, existing benchmarks primarily focus on the accuracy of tool calling --- whether the correct tool is called with the correct parameters --- and less on evaluating *when LMs should (not) call tools*. We develop a new benchmark, **When2Call**, which evaluates tool-calling decision-making: when to generate a tool call, when to ask follow-up questions and when to admit the question can't be answered with the tools provided. and what to do if the question seems to require tool use but a tool call can't be made. We find that state-of-the-art tool-calling LMs show significant room for improvement on **When2Call**, indicating the importance of this benchmark. We also develop a training set for **When2Call** and leverage the multiple-choice nature of the benchmark to develop a preference optimization training regime, which shows considerably more improvement than traditional fine-tuning. This repository contains training and evaluation data along with evaluation code for **When2Call**.


# Dataset

The dataset is also available at: https://huggingface.co/datasets/nvidia/When2Call

## Test

`~/data/test/when2call_test_mcq.jsonl` and `~/data/test/when2call_test_llm_judge.jsonl`

Data format:
- `uuid`: Unique ID for the question
- `source`: Category of the question from BFCL
- `source_id`: ID of the sample from BFCL
- `question`: User question (same as `orig_question` except for "request_for_info" category)
- `orig_question`: Original question (not used during When2Call evaluation)
- `correct_answer`: One of "direct", "tool_call", "request_for_info", or "cannot_answer"
- `answers`: Responses for each of the four answer categories
- `target_tool`: Correct tool to be used for this question (if any)
- `tools`: A list of all tools to be provided with this question (same as `orig_tools` except for "cannot_answer" and "request_for_info" categories)
- `orig_tools`: Original tools for this question provided in the source data (not used during When2Call evaluation)
- `held_out_param`: Name of parameter synthetically removed from user question for "request_for_info" category

```json
{
    "uuid": "1f8221f0-7906-4e11-80f3-6cbe38ce49c8",
    "source": "BFCL v2 Live Simple",
    "source_id": "live_simple_179-104-0",
    "question": "get unique identifier for an ironing service within Chiang Mai?",
    "correct_answer": "cannot_answer",
    "answers": {
        "direct": "The unique identifier for an ironing service in Chiang Mai would be its business registration number or tax ID. You can find this information on their official website or by contacting them directly.",
        "tool_call": "{\"name\": \"get_service_id\", \"arguments\": {\"service_id\": 2, \"province_id\": 2}}",
        "request_for_info": "To assist you better, could you please specify if you're looking for the identifier for a standard ironing service or an extensive ironing service?",
        "cannot_answer": "Apologies, I'm unable to provide that information as I can't perform real-time searches or access databases for specific business details."
    },
    "target_tool": null,
    "tools": [],
    "orig_tools": [
        "{\"name\": \"get_service_id\", \"description\": \"Retrieve the unique identifier for a specific service within a given province.\", \"parameters\": {\"type\": \"dict\", \"required\": [\"service_id\", \"province_id\"], \"properties\": {\"service_id\": {\"type\": \"integer\", \"description\": \"The unique identifier of the service. For example, '1' for cleaning service, '2' for ironing service, and '3' for extensive cleaning service.\", \"enum\": [\"1\", \"2\", \"3\"]}, \"province_id\": {\"type\": \"integer\", \"description\": \"The unique identifier of the province where the service is located. For example, '1' for Bangkok, '2' for Chiang Mai, and '3' for Chonburi.\", \"enum\": [\"1\", \"2\", \"3\"]}}}}"
    ]
}
```

**NOTE**: The LLM-as-a-Judge evaluation data is a subset of the MCQ evaluation data.

## Train

### Supervised Fine-Tuning

`~/data/train/when2call_train_sft.jsonl`

Data format:
- `tools`: A list of tools provided to the model. This list can be converted into a system message
- `messages`: A list of conversation turns, with a question askeed by the *user* and the ground truth response from *assistant*

```json
{
    "tools": [{"name": "get_stations_within_1_km", "description": "Fetch the nearest EV charging stations within a 1 km radius from a given latitude and longitude.", "parameters": {"type": "dict", "properties": {"region": {"description": "The region code (us for United States, ca for Canada, uk for United Kingdom, nz for New Zealand, hk for Hong Kong).", "type": "str", "default": ""}, "latitude": {"description": "The latitude of the location for which to find nearby charging stations.", "type": "int", "default": "40.733"}, "longitude": {"description": "The longitude of the location for which to find nearby charging stations.", "type": "int", "default": "-74.202"}}}, "required": ["region", "latitude", "longitude"]}],
    "messages": [
        {
            "role": "user",
            "content": "What are the trending topics in New York City today?"
        },
        {
            "role": "assistant",
            "content": "Apologies, but I'm unable to provide real-time information or perform web searches. You may want to check a reliable news source for that."
        }
    ]
}
```

### Preference Tuning

`~/data/train/when2call_train_pref.jsonl`

Data format:
- `tools`: A list of tools provided to the model. This list can be converted into a system message
- `messages`: A list containing the question asked by the *user*
- `chosen_response`: An *assistant* response that is correct for the given user question
- `rejected_response`: An incorrect *assistant* response from one of the response categories other than that of the correct response category

```json
{
    "tools": [{"name": "get_company_by_domain", "description": "Fetches company data using a given web domain.", "parameters": {"type": "dict", "properties": {"domain": {"description": "The web domain of the company to look up.", "type": "str"}}}, "required": ["domain"]}],
    "messages": [
        {
            "role": "user",
            "content": "Fetch company details."
        }
    ],
    "chosen_response": {
        "role": "assistant",
        "content": "To provide you with the company details, could you please specify the domain of the company?"
    },
    "rejected_response": {
        "role": "assistant",
        "content": "<TOOLCALL>[{\"name\": \"get_company_by_domain\", \"arguments\": {\"domain\": \"www.apple.com\"}}]</TOOLCALL>"
    }
}
```

**NOTE**: For both SFT and preference datasets, responses with tool calls are enclosed in `<TOOLCALL> ... </TOOLCALL>`, which should be modified according to the prompt template of the model being fine-tuned


# Synthetic Data Generation

- Download the [BFCL benchmark](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/tree/main) data for generating When2Call evaluation data
- Download [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k/tree/main) data for generating When2Call train data
- You will need a key for an OpenAI API compatible endpoint along with the `base_url` and `model` parameters for generating synthetic data.
    - The current dataset was created with https://build.nvidia.com/mistralai/mixtral-8x22b-instruct
    - base_url: https://integrate.api.nvidia.com/v1 and model: `mistralai/mixtral-8x22b-instruct`

## Test

Create evaluation data for MCQ and LLM-as-a-Judge evaluation:
```bash
export OPENAI_API_KEY=<API Key for OpenAI API endpoint>

python ~/synthetic_data_gen/create_eval_data.py \
    --bfcl_data_dir <path to downloaded bfcl data dir> \
    --output_dir <path to dir for saving generated data> \
    --openai_api_base_url <base url for openai api endpoint> \
    --openai_api_model <name of model to use for data generation>
```

## Train

Create raw (unformatted) training data:
```bash
export OPENAI_API_KEY=<API Key for OpenAI API endpoint>

python ~/synthetic_data_gen/create_raw_train_data.py \
    --apigen_data_path <path to downloaded xlam apigen data> \
    --output_dir <path to dir for saving generated data> \
    --openai_api_base_url <base url for openai api endpoint> \
    --openai_api_model <name of model to use for data generation>
```

### SFT

Convert raw training data to SFT format:
```bash
python ~/synthetic_data_gen/convert_raw_train_data_to_sft.py \
    --raw_train_data_path <Path to generated when2call_train_raw.jsonl> \
    --output_dir <path to dir for saving generated data>
```

### Preference

Convert raw training data to Preference Tuning (DPO/RPO) format:
```bash
python ~/synthetic_data_gen/convert_raw_train_data_to_pref.py \
    --raw_train_data_path <Path to generated when2call_train_raw.jsonl> \
    --output_dir <path to dir for saving generated data>
```

# Evaluation

## MCQ (LM-Eval-Harness)

- Clone and set-up [LM-Eval-Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main)
- Copy MCQ evaluation data into `~/evaluation/mcq/lm_eval_harness/when2call`
- Copy `~/evaluation/mcq/lm_eval_harness/when2call` (including the MCQ evaluation data) into `lm-evaluation-harness/lm_eval/tasks`
- Run evaluation as:
```bash
lm_eval \
    --output_path <path to save results> \
    --model hf \
    --model_args pretrained=<model name on huggingface>,parallelize=True \
    --tasks when2call-[functionary|hermes|llama3_2|qwen2_5|xlam] \
    --batch_size <batch size for eval> \
    --num_fewshot 0 \
    --use_cache <path to save model outputs as cache> \
    --log_samples \
    --write_out \
    --trust_remote_code
```
- Generate additional metrics like hallucination rate and confusion matrix for answer categories:
```bash
python `~/evaluation/mcq/lm_eval_harness/when2call/additional_metrics.py` \
    --samples_path <resulting `samples_*.jsonl` file from LM-Eval-Harness>
```
**NOTE**: Please use an appropriate LM-Eval-Harness task name out of `when2call-[functionary|hermes|llama3_2|qwen2_5|xlam]` based on the model that you are evaluating. You can create configurations to accomodate other models with different prompt templates using the `~/evaluation/mcq/lm_eval_harness/when2call/when2call-*.yaml` files as reference

## LLM-as-a-Judge

- Generate model responses using OpenAI API models as:
```bash
python ~/evaluation/llm_as_a_judge/run_openai_inference.py \
    --eval_data_path <path to When2Call eval data for LLM-as-a-Judge eval> \
    --results_path <filepath to save responses> \
    --openai_api_base_url <base url for openai api endpoint> \
    --openai_api_model <name of model to evaluate, ex: gpt-4o-mini>
```
- Run judge using OpenAI API models on model responses as:
```bash
python ~/evaluation/llm_as_a_judge/run_openai_judge.py \
    --responses_path <path to responses generated in previos step> \
    --results_path <filepath to save results> \
    --openai_api_base_url <base url for openai api endpoint> \
    --judge_model <name of model to uses a judge, ex: gpt-4o>
```
- Aggregate LLM-as-a-Judge results:
```bash
python ~/evaluation/llm_as_a_judge/aggregate_llm_as_a_judge_results.py \
    --judge_responses_path <path to judge responses obtained in previous step> \
    --results_path <filepath to save results>
```


# Results

## MCQ

Results on **When2Call**, BFCL Live AST and BFCL Irrelevance for community tool-calling models, and for our Mistral-NeMo-Minitron (MNM) models with and without training on **When2Call** using SFT and Reward-aware Preference Optimization (RPO). For **When2Call**, we show Macro F1, length-normed accuracy and the tool hallucination rate (lower is better $\downarrow$) when no tools are provided. Models not trained on **When2Call** struggle to make the right choices; RPO training yields the greatest benefits.

| Model                     | F1 ↑    | Acc-Norm ↑  | Tool Hall% ↓   | BFCL AST Acc ↑   | BFCL Irr. Acc ↑   |
|--------------------------|----------|-------------|----------------|------------------|-------------------|
| Llama 3.2 1B Instruct    | 21.7     | 45.1%       | 43%            | 13.2%            | 52.9%             |
| Llama 3.2 3B Instruct    | 17.9     | 46.5%       | 52%            | 37.6%            | 46.6%             |
| Llama 3.1 8B Instruct    | 16.6     | 44.2%       | 67%            | 51.6%            | 40.0%             |
| Llama 3.1 70B Instruct   | 37.8     | 46.1%       | 57%            | _68.3%_          | 36.5%             |
| Qwen 2.5 0.5B Instruct   | 32.0     | 53.5%       | 20%            | 22.9%            | 37.7%             |
| Qwen 2.5 1.5B Instruct   | 29.9     | 52.6%       | 23%            | 36.5%            | 71.9%             |
| Qwen 2.5 3B Instruct     | 29.8     | 48.9%       | 23%            | 54.8%            | 53.1%             |
| Qwen 2.5 7B Instruct     | 32.0     | 50.9%       | 21%            | 64.1%            | 51.4%             |
| Qwen 2.5 14B Instruct    | 36.2     | 53.3%       | 21%            | 61.6%            | 64.7%             |
| Qwen 2.5 32B Instruct    | 32.9     | 49.6%       | 17%            | 65.6%            | 63.2%             |
| Qwen 2.5 72B Instruct    | 32.8     | 49.2%       | 23%            | **69.3%**        | 61.1%             |
| xLAM 1B FC-R             | 25.6     | 45.7%       | 40%            | 55.3%            | 61.3%             |
| xLAM 7B FC-R             | 31.5     | 42.7%       | 24%            | 58.3%            | **79.8%**         |
| xLAM 8x7B R              | 32.9     | 47.3%       | 13%            | 67.5%            | 72.4%             |
| xLAM 8x22B R             | 34.3     | 48.3%       | 9.0%           | 74.7%            | 75.2%             |
| MNM 4B SFT (baseline)    | 29.7     | 47.8%       | 16%            | 57.9%            | 41.1%             |
| MNM 4B dataset-SFT       | 48.1     | 67.8%       | 4.3%           | 51.7%            | 67.5%             |
| MNM 4B dataset-RPO       | _51.0_   | _69.1%_     | _1.9%_         | 54.0%            | 77.4%             |
| MNM 8B SFT (baseline)    | 31.9     | 49.1%       | 19%            | 62.2%            | 36.3%             |
| MNM 8B dataset-SFT       | 49.4     | 68.2%       | 7.0%           | 57.5%            | 61.0%             |
| MNM 8B dataset-RPO       | **52.4** | **70.0%**   | **1.2%**       | 62.5%            | _78.1%_           |

## LLM-as-a-Judge

Results on **When2Call**, BFCL v2 Live AST and BFCL Irrelevance for three closed-source tool-calling models using LLM-as-judge evaluation. For BFCL, we report the scores using prompting, not native function-calling, for best comparison.

| Model                  | F1 ↑     | Acc ↑     | Tool Hall% ↓  | BFCL AST Acc ↑   | BFCL Irr. Acc ↑   |
|------------------------|----------|-----------|---------------|------------------|-------------------|
| GPT-4o                 | 61.3     | 61.3%     | 26%           | **79.8%**        | **83.8%**         |
| GPT-4o-Mini            | 52.9     | 54.2%     | 41%           | 76.5%            | 80.7%             |
| GPT-4-Turbo-04-09      | **64.6** | **64.3%** | **22%**       | 63.8%            | 35.6%             |
