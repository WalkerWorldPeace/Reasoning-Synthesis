# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

import torch
import numpy as np
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[str, str], RewardScore]

BatchRewardFunction = Callable[[List[str], List[str]], List[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            ground_truth = data.non_tensor_batch["ground_truth"][i]

            score = self.reward_fn(response_str, ground_truth)
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


# class BatchFunctionRewardManagerRL(FunctionRewardManager):
#     reward_fn: BatchRewardFunction

#     def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
#         response_str, ground_truth, historical_scores = [], [], []
#         response_ids = data.batch["responses"]
#         response_length = data.batch["response_mask"].sum(dim=-1)
#         for i in range(len(data)):
#             valid_response_ids = response_ids[i][: response_length[i]]
#             response_str.append(
#                 self.tokenizer.decode(valid_response_ids, skip_special_tokens=self.config.skip_special_tokens)
#             )
#             ground_truth.append(data.non_tensor_batch["ground_truth"][i])
            
#             # Extract historical score from training data if available
           
#             if "score" in data.non_tensor_batch:
#                 try:
#                     score_str = data.non_tensor_batch["score"][i]
#                     historical_score = float(score_str) if isinstance(score_str, str) else float(score_str)
#                     historical_scores.append(historical_score)
#                 except (ValueError, KeyError) as e:
#                     print(f"Error: Failed to parse score at index {i}: {e}")
#                     print(f"Score value: {data.non_tensor_batch['score'][i] if i < len(data.non_tensor_batch['score']) else 'Index out of range'}")
#                     raise ValueError(f"Invalid score format at index {i}")
#             else:
#                 print("Error: 'score' field not found in data.non_tensor_batch")
#                 print(f"Available keys: {list(data.non_tensor_batch.keys())}")
#                 raise KeyError("Missing required 'score' field in training data")
                    
#         # Try to call reward function with historical scores
#         try:
#             scores = self.reward_fn(response_str, ground_truth, historical_scores=historical_scores)
#         except Exception as e:
#             print(f"Error calling reward function: {e}")
#             print(f"Response count: {len(response_str)}, Ground truth count: {len(ground_truth)}")
#             print(f"Historical scores count: {len(historical_scores)}")
#             raise
            
#         reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
#         reward_metrics = defaultdict(list)
#         for i, score in enumerate(scores):
#             reward_tensor[i, response_length[i] - 1] = score["overall"]
#             for key, value in score.items():
#                 reward_metrics[key].append(value)

#         return reward_tensor, reward_metrics
class BatchFunctionRewardManagerRL(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        response_str, ground_truth, historical_scores, images = [], [], [], []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str.append(
                self.tokenizer.decode(valid_response_ids, skip_special_tokens=self.config.skip_special_tokens)
            )
            ground_truth.append(data.non_tensor_batch["ground_truth"][i])
            
            mm_data = data.non_tensor_batch["multi_modal_data"][i]
            image_list = mm_data["image"]
            first_image = image_list[0]  # PIL.Image 对象
            # 转换为 base64 字符串
            from PIL import Image
            from io import BytesIO
            import base64
            
            if isinstance(first_image, Image.Image):
                buffered = BytesIO()
                first_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                images.append(f"data:image/png;base64,{img_str}")
            else:
                images.append(None)
            
            try:
                score_str = data.non_tensor_batch["score"][i]
                historical_score = float(score_str) if isinstance(score_str, str) else float(score_str)
                historical_scores.append(historical_score)
            except (ValueError, KeyError) as e:
                print(f"Error: Failed to parse score at index {i}: {e}")
                print(f"Score value: {data.non_tensor_batch['score'][i] if i < len(data.non_tensor_batch['score']) else 'Index out of range'}")
                raise ValueError(f"Invalid score format at index {i}")
                    
        scores = self.reward_fn(response_str, ground_truth, images=images, historical_scores=historical_scores)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics