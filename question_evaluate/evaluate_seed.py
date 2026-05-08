import json
import vllm
from transformers import AutoTokenizer
import argparse
import re
import os, sys
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login
import matplotlib.pyplot as plt

# Add root directory to Python path first
root = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(root)
if root not in sys.path:
    sys.path.append(root)
from evaluation.datasets_loader import get_dataset_handler
from mathruler.grader import extract_boxed_content, grade_answer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-Base")
parser.add_argument("--num_samples", type=int, default=10)
parser.add_argument("--dataset", type=str, default="math12k",
                    help="Seed-set nickname. Also used as the output repo name "
                         "(e.g. math12k -> ${HUGGINGFACENAME}/math12k_evaluation).")
parser.add_argument("--source_dataset", type=str, default=None,
                    help="HF repo id of the source dataset to evaluate. "
                         "If omitted, resolved from --dataset via the built-in registry.")
parser.add_argument("--config", type=str, default=None,
                    help="Optional HF dataset config name (e.g. 'en' for DAPO, 'main' for GSM8K).")
parser.add_argument("--question_field", type=str, default=None,
                    help="Field name holding the question text. Defaults from registry.")
parser.add_argument("--answer_field", type=str, default=None,
                    help="Field name holding the reference answer. Defaults from registry.")
parser.add_argument("--splits", type=str, default=None,
                    help="Comma-separated splits to evaluate (e.g. 'train,test'). "
                         "Defaults from registry. Missing splits are skipped with a warning.")
args = parser.parse_args()
STORAGE_PATH = os.getenv("STORAGE_PATH")


# Registry: nickname -> (source repo id, config, question field, answer field, default splits)
DATASET_REGISTRY = {
    "math12k": ("hiyouga/math12k",                 None,   "problem",  "answer",   ["train"]),
    "dapo":    ("open-r1/DAPO-Math-17k-Processed", "en",   "prompt",   "solution", ["train", "test"]),
    "gsm8k":   ("openai/gsm8k",                    "main", "question", "answer",   ["train", "test"]),
}


def resolve_source_config():
    """Resolve source repo / config / field names / splits for the requested seed set.

    Priority: explicit CLI flags > registry entry (by --dataset nickname).
    Raises if the user gave a nickname we do not know and did not supply --source_dataset.
    """
    entry = DATASET_REGISTRY.get(args.dataset.lower())
    if args.source_dataset is None and entry is None:
        raise ValueError(
            f"Unknown --dataset nickname '{args.dataset}'. "
            f"Either pick one of {sorted(DATASET_REGISTRY.keys())} "
            f"or pass --source_dataset <hf_repo_id> --question_field ... --answer_field ...."
        )
    reg_src, reg_cfg, reg_qf, reg_af, reg_splits = entry if entry is not None else (None, None, None, None, ["train"])
    src = args.source_dataset or reg_src
    cfg = args.config          or reg_cfg
    qf  = args.question_field  or reg_qf  or "problem"
    af  = args.answer_field    or reg_af  or "answer"
    splits = [s.strip() for s in args.splits.split(",")] if args.splits else reg_splits
    return src, cfg, qf, af, splits

def extract_answer(response):
    match = re.search(r"\\boxed{(.*?)}", response)
    return match.group(1) if match else None

def process_split(split_name, questions, answers):
    """Run majority-vote evaluation on one split; return (results, accuracy)."""
    print(f"Processing {split_name} split with {len(questions)} questions...")

    # Create chat format for all questions
    chats = [[{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},{"role": "user", "content": question}] for question in questions]
    if tokenizer.chat_template:
        prompts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, add_special_tokens=True) for chat in chats]
    else:
        prompts = ["system: " + chat[0]["content"] + '\n' + "user: " + chat[1]["content"] for chat in chats]

    # Generate responses
    print(f"Generating responses for {split_name}...")
    responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)

    print(f"Processing {len(responses)} responses for {split_name}...")
    results_all = []

    for response, answer, question in zip(responses, answers, questions):
        try:
            error_flag = False
            results = [extract_boxed_content(output.text) for output in response.outputs]
            results = [r for r in results if r is not None and str(r).strip() not in ["", "None"]]

            if not results:
                continue

            answer_counts = {}
            for result in results:
                found_match = False
                try:
                    for existing_answer in answer_counts:
                        if grade_answer(result, existing_answer) or grade_answer(existing_answer, result) or result == existing_answer or ('no ' in result.lower() and 'no ' in existing_answer.lower()):
                            answer_counts[existing_answer] += 1
                            found_match = True
                            break
                except Exception:
                    error_flag = True
                    break
                if not found_match:
                    answer_counts[result] = 1

            if error_flag:
                continue
            if answer_counts:
                max_count = max(answer_counts.values())
                majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
                score = max_count / len(results)

                # Skip proof / box-wrapping / text-heavy items that the rule grader cannot handle.
                if "证明" in question or 'box' in question.lower() or 'text' in majority_answer.lower():
                    continue

                results_all.append({
                    "question": question,
                    "answer": majority_answer,
                    "score": score,
                    'results': results,
                    'ground_truth': answer
                })
        except Exception as e:
            print("Error:", e)
            continue

    print(f"Successfully processed {len(results_all)} questions for {split_name}")

    # Calculate accuracy for this split
    correct_list = []
    for data in results_all:
        if grade_answer(data['answer'], data['ground_truth']) or grade_answer(data['ground_truth'], data['answer']) or data['answer'] == data['ground_truth'] or ('no ' in data['answer'].lower() and 'no ' in data['ground_truth'].lower()):
            correct = 1
        else:
            correct = 0
        correct_list.append(correct)
    average_accuracy = sum(correct_list) / len(correct_list) if correct_list else 0.0
    print(f"{split_name} majority answer exact match accuracy: {average_accuracy:.4f}")

    return results_all, average_accuracy

print('Loading dataset...')
source_repo, config, qfield, afield, splits = resolve_source_config()
print(f"Source dataset: {source_repo}" + (f" (config={config})" if config else "")
      + f"  fields: question='{qfield}', answer='{afield}'  splits={splits}")


def _try_load(split):
    try:
        return load_dataset(source_repo, config, split=split) if config else load_dataset(source_repo, split=split)
    except Exception as e:
        print(f"Warning: could not load split '{split}' from {source_repo}: {e}")
        return None


train_dataset = _try_load("train") if "train" in splits else None
test_dataset  = _try_load("test")  if "test"  in splits else None

train_examples = [row for row in train_dataset] if train_dataset is not None else []
test_examples  = [row for row in test_dataset]  if test_dataset  is not None else []

train_questions = [example[qfield] for example in train_examples]
train_answers   = [example[afield] for example in train_examples]

test_questions  = [example[qfield] for example in test_examples]
test_answers    = [example[afield] for example in test_examples]

print(f'Loaded {len(train_questions)} train questions and {len(test_questions)} test questions from {source_repo}')

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = vllm.LLM(
    model=args.model,
    tokenizer=args.model,
    tensor_parallel_size=8,  # Use 4 GPUs for parallel inference
    gpu_memory_utilization=0.85,
    seed=77,  # Fixed seed for reproducibility
)
sample_params = vllm.SamplingParams(
    max_tokens=8192,
    temperature=0.6,
    top_p=0.95,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=args.num_samples,
)

# Process both splits (skip splits that were not requested / not available)
if train_questions:
    train_results, train_accuracy = process_split("train", train_questions, train_answers)
else:
    train_results, train_accuracy = [], 0.0
if test_questions:
    test_results, test_accuracy = process_split("test", test_questions, test_answers)
else:
    test_results, test_accuracy = [], 0.0

# Combine results for overall statistics
all_results = train_results + test_results
overall_accuracy = (train_accuracy * len(train_results) + test_accuracy * len(test_results)) / (len(train_results) + len(test_results)) if (len(train_results) + len(test_results)) > 0 else 0.0

print(f"\n=== Overall Results ===")
print(f"Train accuracy: {train_accuracy:.4f} ({len(train_results)} questions)")
print(f"Test accuracy: {test_accuracy:.4f} ({len(test_results)} questions)")
print(f"Overall accuracy: {overall_accuracy:.4f} ({len(all_results)} questions)")

# Final results with accuracy summary at the top
final_results = [{
    "train_accuracy": train_accuracy,
    "test_accuracy": test_accuracy,
    "overall_accuracy": overall_accuracy,
    "train_questions_count": len(train_results),
    "test_questions_count": len(test_results),
    "total_questions_count": len(all_results)
}] + train_results + test_results

local_json_path = f"{args.dataset}_majority_results.json"
with open(local_json_path, "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)
print(f"Local results saved to {local_json_path}")

# Direct upload to Hugging Face Hub
print("Preparing data for upload...")
# Load Hugging Face token and login
HUGGINGFACENAME = os.getenv("HUGGINGFACENAME")
try:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token is None and os.path.exists("tokens.json"):
        with open("tokens.json", "r") as f:
            token = json.load(f).get("huggingface")
    if token:
        login(token=token)
    else:
        print("Warning: no HF_TOKEN env var and no tokens.json; skipping HF login.")
except Exception as e:
    print(f"Warning: Could not login to Hugging Face: {e}")

# Extract scores for distribution analysis
all_scores = [data['score'] for data in all_results]
train_scores = [data['score'] for data in train_results]
test_scores = [data['score'] for data in test_results]

if all_scores:
    print(f"Overall score distribution: min={min(all_scores):.3f}, max={max(all_scores):.3f}, mean={sum(all_scores)/len(all_scores):.3f}")

# Create and save score distribution plot
if all_scores:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Overall distribution
    axes[0].hist(all_scores, bins=20)
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Overall Score Distribution ({len(all_scores)} questions)')
    
    # Train distribution
    if train_scores:
        axes[1].hist(train_scores, bins=20, alpha=0.7, color='blue')
        axes[1].set_xlabel('Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Train Score Distribution ({len(train_scores)} questions)')
    
    # Test distribution
    if test_scores:
        axes[2].hist(test_scores, bins=20, alpha=0.7, color='red')
        axes[2].set_xlabel('Score')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title(f'Test Score Distribution ({len(test_scores)} questions)')
    
    plt.tight_layout()
    plt.savefig('scores_distribution.png')
    plt.close()
    print("Score distribution plot saved as scores_distribution.png")

# Prepare data for upload
train_filtered_data = [
    {
        'problem': data['question'],
        'answer': data['answer'], 
        'score': data['score']
    } 
    for data in train_results 
    if data['answer'] != '' and data['answer'] != 'None'
]

test_filtered_data = [
    {
        'problem': data['question'],
        'answer': data['answer'], 
        'score': data['score']
    } 
    for data in test_results 
    if data['answer'] != '' and data['answer'] != 'None'
]

print(f"Prepared {len(train_filtered_data)} train questions and {len(test_filtered_data)} test questions for upload")

# Upload to Hugging Face Hub if we have data
if (train_filtered_data or test_filtered_data) and HUGGINGFACENAME:
    try:
        repo_name = f"{args.dataset}_evaluation"
        
        dataset_dict = {}
        if train_filtered_data:
            dataset_dict["train"] = Dataset.from_list(train_filtered_data)
        if test_filtered_data:
            dataset_dict["test"] = Dataset.from_list(test_filtered_data)
        
        dataset = DatasetDict(dataset_dict)
        
        print(f"Uploading to {HUGGINGFACENAME}/{repo_name}")
        dataset.push_to_hub(f"{HUGGINGFACENAME}/{repo_name}", private=True)
        print(f"Successfully uploaded {len(train_filtered_data)} train and {len(test_filtered_data)} test questions to Hugging Face Hub")
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {e}")
else:
    if not (train_filtered_data or test_filtered_data):
        print("No data to upload after filtering")
    if not HUGGINGFACENAME:
        print("HUGGINGFACENAME environment variable not set, skipping upload")

print("Processing completed!")