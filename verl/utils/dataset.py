# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file exompliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF

import json
import random
def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}



def process_image(image: Union[Dict[str, Any], ImageObject, str], min_pixels: int, max_pixels: int) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    # ✅ 先处理调色板模式的透明度
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_prompts = filter_overlong_prompts

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            self.dataset = load_dataset("parquet", data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("parquet", data_files=data_path, split=data_split)
        elif "openai/gsm8k" in data_path:
            # load gsm8k dataset
            self.dataset = load_dataset(data_path, "main", split=data_split)
        elif "open-r1/DAPO-Math-17k-Processed" in data_path:
            self.dataset = load_dataset(data_path, "en", split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if "questioner_format_with_persona" in self.format_prompt:
            print("load personas")
            personas_dataset = load_dataset("proj-persona/PersonaHub", "math", split="train")
            self.personas = [item['input persona'] for item in personas_dataset]
            # self.personas = self.personas.select(range(100))
        if self.filter_overlong_prompts:
            self.dataset = self.dataset.filter(self._filter_overlong_prompts, desc="Filtering overlong prompts")
   
    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            prompt_str: str = example[self.prompt_key]
            answer_str: str = example[self.answer_key]
        except KeyError:
            if "question" in example:
                prompt_str: str = example["question"]
            elif "problem" in example:
                prompt_str: str = example["problem"]
            else:
                raise KeyError(f"Cannot find prompt key '{self.prompt_key}' in the dataset example.")
        if "questioner_format" in self.format_prompt:
            # system_prompt = (
            #     "You are a senior mathematics problem setter. Your task is to generate a new 'Problem 2' based on the given 'Problem 1'.\n"
            #     "Output format requirements:\n"
            #     "- Put the new problem within <question></question> tags\n"
            # )
            # formatted_prompt = f"Please create a new problem based on: <question>{prompt_str}</question>."
            # return [{"role": "system", "content": system_prompt},{"role": "user", "content": formatted_prompt}]
            formatted_prompt = f"Please create a new problem based on: <question>{prompt_str}</question>. Please reason step by step inside <think>...</think> and output only the final problem inside <question>...</question>."
            return [{"role": "user", "content": formatted_prompt}]
        
        if "questioner_rl_format" in self.format_prompt:
            accuracy_str: str = example["score"]
            # system_prompt = (
            #     "You are an expert mathematics problem setter. "
            #     "Your task is to generate a new mathematical problem (Problem 2) based on the given reference problem (Problem 1) "
            #     "and the student's current accuracy rate. Apply the following difficulty adjustment rules:\n\n"
            #     "Difficulty Adjustment Guidelines:\n"
            #     "- If accuracy < 0.3 (low): Simplify the problem significantly - reduce complexity or break down into simpler steps\n"
            #     "- If 0.3 ≤ accuracy ≤ 0.7 (medium): Maintain similar difficulty level\n"
            #     "- If accuracy > 0.7 (high): Increase difficulty - add complexity, introduce additional constraints, or combine multiple concepts\n\n"
            #     "Output format requirements:\n"
            #     "- Enclose the new problem within <question></question> tags\n"
            # )
            # formatted_prompt = f"Based on the reference problem: <question>{prompt_str}</question>\nStudent's current accuracy rate: {accuracy_str}\n\nPlease create a new mathematical problem with appropriate difficulty adjustment."
            # return [{"role": "system", "content": system_prompt},{"role": "user", "content": formatted_prompt}]
            formatted_prompt = f"Please create a novel self-contained problem with appropriate difficulty adjustment based on: <question>{prompt_str}</question> and student's current accuracy rate: {accuracy_str}. Apply the following difficulty adjustment rules: If accuracy < 0.3 (low): Simplify the problem significantly - reduce complexity or break down into simpler steps. If 0.3 ≤ accuracy ≤ 0.7 (medium): Maintain similar difficulty level. If accuracy > 0.7 (high): Increase difficulty - add complexity, introduce additional constraints, or combine multiple concepts. Please reason step by step inside <think>...</think> and output only the final problem inside <question>...</question>."
            return [{"role": "user", "content": formatted_prompt}]
        if "answer_format" in self.format_prompt:
            system_prompt = (
                "You are a senior mathematics problem setter. Your task is to generate a new 'Problem 2' and provide its 'Answer 2' based on the given 'Problem 1' and its 'Answer 1'.\n"
                "Output format requirements:\n"
                "- Put the new problem within <question></question> tags\n"
                r"- Put your final answer within \boxed{}."
            )
            formatted_prompt = f"Please create a new problem based on: <question>{prompt_str}</question> and <answer>{answer_str}</answer>. Remember to format the output exactly as instructed."
            
            return [{"role": "system", "content": system_prompt},{"role": "user", "content": formatted_prompt}]
        if "solver_format" in self.format_prompt:
            return [{"role": "system", "content": r"Please reason step by step, and put your final answer within \boxed{}."},{"role": "user", "content": prompt_str}]
        
        if "questioner_rlvl_format" in self.format_prompt:
            accuracy_str: str = example["score"]
            
            # 检查是否有图片
            if self.image_key in example and example[self.image_key]:
                # 构建多模态内容
                content_list = []
                
                # 添加图片
                images = example[self.image_key]
                for _ in images:
                    content_list.append({"type": "image"})
                
                # 添加文本提示
                formatted_prompt = (
                    f"Please create a novel self-contained problem with appropriate difficulty adjustment "
                    f"based on the given reference problem and student's current accuracy rate: {accuracy_str}. "
                    f"The reference problem is: <question>{prompt_str}</question>. "
                    f"Apply the following difficulty adjustment rules: "
                    f"If accuracy < 0.3 (low): Simplify the problem significantly - reduce complexity or break down into simpler steps. "
                    f"If 0.3 ≤ accuracy ≤ 0.7 (medium): Maintain similar difficulty level. "
                    f"If accuracy > 0.7 (high): Increase difficulty - add complexity, introduce additional constraints, or combine multiple concepts. "
                    f"Please reason step by step inside <think>...</think> and output only the final problem inside <question>...</question>."
                )
                # formatted_prompt = f"Please create a new problem based on: <question>{prompt_str}</question>. Please reason step by step inside <think>...</think> and output only the final problem inside <question>...</question>."
                content_list.append({"type": "text", "text": formatted_prompt})
                
                return [{"role": "user", "content": content_list}]
        
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]
        
    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            raw_image_data = example.pop(self.image_key)
            images = [
                process_image(image, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
                for image in raw_image_data
            ]
            model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"image": raw_image_data}
        else:
            if self.tokenizer.chat_template:
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                prompt = "system: " + messages[0]["content"] + '\n' + "user: " + messages[1]["content"]
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)
        # if self.image_key in example:
        #     prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        #     images = example.pop(self.image_key)
        #     if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
        #         images = [os.path.join(self.image_dir, image) for image in images]

        #     processed_images = [] if len(images) != 0 else None  # text-only data
        #     for image in images:
        #         processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

        #     model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
        #     input_ids = model_inputs.pop("input_ids")[0]
        #     attention_mask = model_inputs.pop("attention_mask")[0]
        #     example["multi_modal_data"] = {"images": images}
        # elif self.video_key in example:
        #     prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        #     videos = example.pop(self.video_key)
        #     if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
        #         videos = [os.path.join(self.image_dir, video) for video in videos]

        #     processed_videos = [] if len(videos) != 0 else None  # text-only data
        #     video_fps_list = []
        #     for video in videos:
        #         processed_video, video_fps = process_video(
        #             video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
        #         )
        #         processed_videos.append(processed_video)
        #         video_fps_list.append(video_fps)

        #     model_inputs = self.processor(
        #         videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
        #     )
        #     if "second_per_grid_ts" in self.processor.model_input_names:
        #         model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

        #     input_ids = model_inputs.pop("input_ids")[0]
        #     attention_mask = model_inputs.pop("attention_mask")[0]
        #     example["multi_modal_data"] = {"videos": videos}
        # else:
        #     prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        #     model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
        #     input_ids = model_inputs.pop("input_ids")[0]
        #     attention_mask = model_inputs.pop("attention_mask")[0]

        # if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
        #     # qwen2vl mrope
        #     position_ids = get_rope_index(
        #         self.processor,
        #         input_ids=input_ids,
        #         image_grid_thw=model_inputs.get("image_grid_thw", None),
        #         video_grid_thw=model_inputs.get("video_grid_thw", None),
        #         second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
        #         attention_mask=attention_mask,
        #     )  # (3, seq_length)
        # else:
        #     position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
            
        return example
