import vllm
from transformers import AutoTokenizer
import argparse
from typing import List
from vllm.outputs import RequestOutput
import json
import regex as re
import os, sys

root = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(root)
if root not in sys.path:
    sys.path.append(root)

from datasets import load_dataset
import random
STORAGE_PATH = os.getenv("STORAGE_PATH")


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        max_model_len=16384,
        seed=int(args.suffix),
    )

    dataset = load_dataset(f"{os.environ['HUGGINGFACENAME']}/math12k_evaluation", split="train")
    questions = [item["problem"] for item in dataset]
    answers = [item["answer"] for item in dataset]
    scores = [item["score"] for item in dataset]
    print(f"Loaded MATH dataset with {len(questions)} questions")
    title_random = random.Random(int(args.suffix))
    if args.num_samples > len(questions):
        selected_indices = title_random.choices(range(len(questions)), k=args.num_samples)
    else:
        selected_indices = title_random.sample(range(len(questions)), args.num_samples)

    prompts = []
    for i in range(args.num_samples):
        ref_idx = selected_indices[i]
        ref_question = questions[ref_idx]
        ref_score = scores[ref_idx]

        chat = [
            {
                "role": "user",
                "content": f"Please create a novel self-contained problem with appropriate difficulty adjustment based on: <question>{ref_question}</question> and student's current accuracy rate: {ref_score}. Apply the following difficulty adjustment rules: If accuracy < 0.3 (low): Simplify the problem significantly - reduce complexity or break down into simpler steps. If 0.3 ≤ accuracy ≤ 0.7 (medium): Maintain similar difficulty level. If accuracy > 0.7 (high): Increase difficulty - add complexity, introduce additional constraints, or combine multiple concepts. Please reason step by step inside <think>...</think> and output only the final problem inside <question>...</question>."
            }
        ]

        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=True,
            )
        else:
            prompt = "user: " + chat[0]["content"]

        prompts.append(prompt)

    sample_params = vllm.SamplingParams(
        max_tokens=8192,
        temperature=1.0,
        top_p=0.95,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    completions: List[RequestOutput] = model.generate(prompts, sampling_params=sample_params)

    results = []
    for completion in completions:
        response = completion.outputs[0].text
        try:
            questions = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
            if questions:
                results.append({"question": questions[-1].strip(), "answer": "", "score": 0})
        except Exception:
            results.append({"question": response, "answer": "", "score": -1})
    with open(f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--num_samples", type=int, default=1250, help="Number of samples to generate")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add to the output file")
    parser.add_argument("--save_name", type=str, default="")
    args = parser.parse_args()

    main(args)