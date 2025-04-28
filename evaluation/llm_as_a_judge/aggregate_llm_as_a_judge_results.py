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

from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge_responses_path",
        type=str,
        required=True,
        help="Path to judge responses.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Filepath to save where results.",
    )

    args = parser.parse_args()

    data = []
    with open(args.judge_responses_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    categories = {
        "tool_call": {"score": [], "response_distribution": []},
        "request_for_info": {"score": [], "response_distribution": []},
        "cannot_answer": {"score": [], "response_distribution": []},
    }

    for item in data:
        categories[item["question"]["correct_answer"]]["score"].append(int(item["judge_response"]["classification"] == item["question"]["correct_answer"]))
        categories[item["question"]["correct_answer"]]["response_distribution"].append(item["judge_response"]["classification"])

    for category in categories:
        assert len(categories[category]["score"]) == len(categories[category]["response_distribution"])
        categories[category]["score"] = sum(categories[category]["score"]) / len(categories[category]["score"])
        categories[category]["response_distribution"] = dict(Counter(categories[category]["response_distribution"]))

    print(categories)


if __name__ == "__main__":
    main()
