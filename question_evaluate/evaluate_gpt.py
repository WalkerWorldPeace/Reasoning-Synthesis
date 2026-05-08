#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Label synthetic math problems with a strong reasoning model (Sec. 4.3).

For every generated problem, we ask a reasoning model (default `o3-mini`) to
either produce a final answer enclosed in ``\\boxed{...}`` or say `no solution`.
Problems labeled "no solution" get score=0 and are later filtered out by
``combine_datasets.py``; every other problem is kept with the model's answer.

The model is called through any OpenAI-compatible Chat Completions endpoint.
Configure it via environment variables:

  OPENAI_API_KEY    API key for your provider
  OPENAI_BASE_URL   (optional) base URL, defaults to https://api.openai.com/v1
                    e.g. https://openrouter.ai/api/v1 for OpenRouter,
                         https://generativelanguage.googleapis.com/v1beta/openai
                         for Gemini via its OpenAI-compatible endpoint, etc.

Pass the model name with ``--gpt_model``.

Example:
    OPENAI_API_KEY=sk-... python question_evaluate/evaluate_gpt.py \\
        --gpt_model o3-mini --suffix 0 --save_name my_experiment
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime, timedelta
from tqdm import tqdm

root = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(root)
if root not in sys.path:
    sys.path.append(root)

from mathruler.grader import extract_boxed_content  # noqa: E402

try:
    from openai import OpenAI
except ImportError as exc:
    raise RuntimeError(
        "The OpenAI SDK is required. Install it with `pip install openai`."
    ) from exc


# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Label synthesized math problems via an OpenAI-compatible API.",
)
parser.add_argument("--gpt_model", type=str, default="o3-mini",
                    help="Model name to call (default: o3-mini).")
parser.add_argument("--suffix", type=str, default="0",
                    help="Shard id; usually the GPU / process index (0..7).")
parser.add_argument("--save_name", type=str, required=True,
                    help="Experiment name used to form input/output file paths.")
parser.add_argument("--max_retries", type=int, default=3,
                    help="Max retries per API call.")
parser.add_argument("--retry_delay", type=float, default=1.0,
                    help="Base delay (seconds) between retries (doubled each attempt).")
parser.add_argument("--rate_limit", type=int, default=12,
                    help="Maximum requests per minute per process.")
parser.add_argument("--max_tokens", type=int, default=4096,
                    help="Max tokens of the model reply.")
args = parser.parse_args()

# --- Paths ---
STORAGE_PATH = os.getenv("STORAGE_PATH")
if STORAGE_PATH is None:
    raise RuntimeError("STORAGE_PATH env var is not set")
INPUT_FILE = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json"
OUTPUT_FILE = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}_results.json"


# --- Rate Limiting ---
class RateLimiter:
    def __init__(self, max_requests_per_minute):
        self.max_requests = max_requests_per_minute
        self.requests = queue.Queue()
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            now = datetime.now()
            temp_queue = queue.Queue()
            while not self.requests.empty():
                request_time = self.requests.get()
                if now - request_time < timedelta(minutes=1):
                    temp_queue.put(request_time)
            self.requests = temp_queue

            if self.requests.qsize() >= self.max_requests:
                oldest = self.requests.queue[0]
                wait_time = 60 - (now - oldest).total_seconds()
                if wait_time > 0:
                    print(f"[{args.suffix}] Rate limit reached; sleeping {wait_time:.1f}s")
                    time.sleep(wait_time)

            self.requests.put(now)


rate_limiter = RateLimiter(args.rate_limit)


# --- Client ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Provide an API key for any OpenAI-compatible "
        "endpoint (OpenAI, OpenRouter, Gemini's OpenAI-compatible endpoint, etc.)."
    )
client = OpenAI(
    api_key=api_key,
    base_url=os.getenv("OPENAI_BASE_URL"),  # None => OpenAI default
)

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}. "
    "If the problem has no solution, please output 'no solution' in the box."
)


def generate_single_answer(question: str, model: str):
    """Call the model once; return the assistant text or None on failure."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for attempt in range(args.max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=args.max_tokens,
            )
            content = resp.choices[0].message.content
            if content:
                return content.strip()
        except Exception as e:
            print(f"[{args.suffix}] API call failed on attempt {attempt + 1}/{args.max_retries}: {e}")
        if attempt < args.max_retries - 1:
            wait = args.retry_delay * (2 ** attempt)
            time.sleep(wait)
    return None


# --- Load synthesized problems ---
print(f"[{args.suffix}] Loading data from: {INPUT_FILE}")
try:
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"[{args.suffix}] ERROR: input file not found. Exiting.")
    sys.exit(1)

# Filter out rows that the generator already flagged as invalid.
correct_data = [item for item in data if item.get("score", 0) >= 0]
if not correct_data:
    print(f"[{args.suffix}] Nothing to label; writing empty results.")
    with open(OUTPUT_FILE, "w") as f:
        json.dump([], f)
    sys.exit(0)

questions = [item["question"] for item in correct_data]

print(f"[{args.suffix}] Found {len(questions)} questions; rate limit = "
      f"{args.rate_limit}/min; model = {args.gpt_model}")

# --- Call the API ---
all_responses = []
for i, question in enumerate(tqdm(questions, desc=f"labeling[{args.suffix}]")):
    rate_limiter.wait_if_needed()
    resp = generate_single_answer(question, args.gpt_model)
    all_responses.append([resp] if resp else [])
    if (i + 1) % 10 == 0:
        ok = sum(1 for r in all_responses if r) / len(all_responses) * 100
        print(f"[{args.suffix}] {i + 1}/{len(questions)} - current success rate {ok:.1f}%")

# --- Score the replies ---
results_all = []
for responses, question in zip(all_responses, questions):
    if not responses:
        continue
    response = responses[0]
    boxed = extract_boxed_content(response)
    if not boxed or not boxed.strip():
        continue
    result = boxed.strip()

    # score=0 <=> "no solution"; score=1 otherwise.
    lowered = result.lower()
    is_no_solution = (
        "no solution" in lowered
        or "no answer" in lowered
        or "no\\ solution" in result
        or lowered == "none"
        or result == "None"
    )
    score = 0 if is_no_solution else 1

    # Skip proof-style questions that cannot be auto-graded.
    if "证明" in question or "box" in question.lower():
        continue

    results_all.append({
        "question": question,
        "answer": result,
        "score": score,
        "gpt_model_used": args.gpt_model,
        "response_text": response,
        "is_no_solution": is_no_solution,
    })

# --- Save ---
with open(OUTPUT_FILE, "w") as f:
    json.dump(results_all, f, indent=4, ensure_ascii=False)

print(f"[{args.suffix}] done; wrote {len(results_all)} rows to {OUTPUT_FILE}")
if results_all:
    no_solution = sum(1 for x in results_all if x["score"] == 0)
    has_solution = len(results_all) - no_solution
    print(f"[{args.suffix}] has_solution={has_solution}, no_solution={no_solution}")
