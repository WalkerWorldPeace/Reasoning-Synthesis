import vllm
from transformers import AutoTokenizer, AutoProcessor
import argparse
from typing import List
from vllm.outputs import RequestOutput
import json
import regex as re
import os, sys
from PIL import Image
from io import BytesIO
import base64

root = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(root)
if root not in sys.path:
    sys.path.append(root)

from datasets import load_dataset
import random
STORAGE_PATH = os.getenv("STORAGE_PATH")
if STORAGE_PATH is None:
    raise RuntimeError("STORAGE_PATH env var is not set")
HUGGINGFACENAME = os.getenv("HUGGINGFACENAME")
if HUGGINGFACENAME is None:
    raise RuntimeError("HUGGINGFACENAME env var is not set")


def process_image(image, min_pixels=None, max_pixels=None):
    """Load an image (path / {'bytes': ...} / raw bytes / PIL), resize to fit
    [min_pixels, max_pixels], and convert to RGB."""
    import math

    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()

    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def image_to_base64(pil_image):
    """Encode a PIL image as a base64 PNG string."""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def main(args):
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        max_model_len=8192,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.85,
        seed=int(args.suffix),
    )

    # VL seed-evaluation repo produced by question_evaluate/evaluate_seed_vl.py
    # carries `score` = a_ori (solver consistency on that problem).
    seed_repo = os.getenv("VL_SEED_DATASET", f"{HUGGINGFACENAME}/mmk12_vl_evaluation")
    print(f"Loading {seed_repo}...")
    dataset = load_dataset(seed_repo, split="train")
    questions = [item["problem"] for item in dataset]
    images_list = [item["images"] for item in dataset]
    scores = [item["score"] for item in dataset]
    print(f"Loaded {len(questions)} questions")

    title_random = random.Random(int(args.suffix))
    if args.num_samples > len(questions):
        selected_indices = title_random.choices(range(len(questions)), k=args.num_samples)
    else:
        selected_indices = title_random.sample(range(len(questions)), args.num_samples)

    prompts_with_images = []
    for i in range(args.num_samples):
        ref_idx = selected_indices[i]
        ref_question = questions[ref_idx]
        ref_images = images_list[ref_idx]
        ref_score = scores[ref_idx]

        processed_images = []
        if ref_images:
            for img in ref_images:
                try:
                    processed_images.append(process_image(
                        img, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
                    ))
                except Exception as e:
                    print(f"Error processing image in sample {i}: {e}")

        content_list = [{"type": "image"} for _ in processed_images]
        content_list.append({
            "type": "text",
            "text": (
                f"Please create a novel self-contained problem with appropriate difficulty adjustment "
                f"based on the given reference problem and student's current accuracy rate: {ref_score}. "
                f"The reference problem is: <question>{ref_question}</question>. "
                f"Apply the following difficulty adjustment rules: "
                f"If accuracy < 0.3 (low): Simplify the problem significantly - reduce complexity or break down into simpler steps. "
                f"If 0.3 ≤ accuracy ≤ 0.7 (medium): Maintain similar difficulty level. "
                f"If accuracy > 0.7 (high): Increase difficulty - add complexity, introduce additional constraints, or combine multiple concepts. "
                f"Please reason step by step inside <think>...</think> and output only the final problem inside <question>...</question>."
            ),
        })

        chat = [{"role": "user", "content": content_list}]
        prompt = processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, add_special_tokens=True,
        )
        prompts_with_images.append({
            "prompt": prompt,
            "multi_modal_data": {"image": processed_images},
        })

    sample_params = vllm.SamplingParams(
        max_tokens=4096, temperature=1.0, top_p=0.95, top_k=50, n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    print(f"Generating {len(prompts_with_images)} samples...")
    completions: List[RequestOutput] = model.generate(
        prompts_with_images, sampling_params=sample_params, use_tqdm=True,
    )
    print(f"Successfully generated {len(completions)} completions")

    results = []
    for i, completion in enumerate(completions):
        response = completion.outputs[0].text
        try:
            questions_found = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
            if not questions_found:
                continue
            question = questions_found[-1].strip()
            ref_idx = selected_indices[i]
            ref_images = images_list[ref_idx]

            image_base64 = None
            if ref_images:
                img = ref_images[0]
                if isinstance(img, dict):
                    pil_img = Image.open(BytesIO(img["bytes"]))
                elif isinstance(img, Image.Image):
                    pil_img = img
                else:
                    pil_img = process_image(img)
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                image_base64 = image_to_base64(pil_img)

            results.append({
                "question": question,
                "answer": "",
                "score": 0,
                "image": image_base64,
            })
        except Exception as e:
            print(f"Error extracting question: {e}")
            results.append({"question": response, "answer": "", "score": -1, "image": None})

    output_path = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Generated {len(results)} questions and saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--num_samples", type=int, default=1250, help="Number of samples to generate")
    parser.add_argument("--suffix", type=str, default="0", help="Suffix to add to the output file")
    parser.add_argument("--save_name", type=str, default="geometry_questions", help="Output filename prefix")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    args = parser.parse_args()

    main(args)