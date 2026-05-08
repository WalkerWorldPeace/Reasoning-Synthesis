import json
import vllm
from transformers import AutoTokenizer
import argparse
import re
import os, sys
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

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
parser.add_argument("--dataset", type=str, default="math12k", help="Dataset name to evaluate")
args = parser.parse_args()
STORAGE_PATH = os.getenv("STORAGE_PATH")

def extract_answer(response):
    match = re.search(r"\\boxed{(.*?)}", response)
    return match.group(1) if match else None

def process_split(split_name, questions, answers):
    """处理单个数据分割（train或test）"""
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
            count = 0
            results = [extract_boxed_content(output.text) for output in response.outputs]
            results = [r for r in results if r is not None and str(r).strip() not in ["", "None"]]
            
            # 如果过滤后没有有效结果，跳过这个问题
            if not results:
                continue
            
            answer_counts = {}
            for result in results:
                found_match = False
                # Check if result matches any existing answer group
                try:
                    for existing_answer in answer_counts:
                        if grade_answer(result, existing_answer) or grade_answer(existing_answer, result) or result == existing_answer or ('no ' in result.lower() and 'no ' in existing_answer.lower()):
                            answer_counts[existing_answer] += 1
                            found_match = True
                            break
                except Exception:
                    error_flag = True
                    break
                # If no match found, create new answer group
                if not found_match:
                    answer_counts[result] = 1

            # Find the answer with the most matches
            if error_flag:
                continue
            if answer_counts:
                max_count = max(answer_counts.values())
                majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
                score = max_count / len(results)
                
                # Skip certain types of questions
                if "证明" in question or 'box' in question.lower() or 'text' in majority_answer.lower():
                    continue
                
                # Calculate accuracy for this question
                correct_count = 0
                for result in results:
                    try:
                        if grade_answer(result, answer) or grade_answer(answer, result) or result == answer or ('no ' in result.lower() and 'no ' in answer.lower()):
                            correct_count += 1
                    except Exception:
                        pass
                accuracy = correct_count / len(results) if results else 0.0
                    
                results_all.append({
                    "question": question, 
                    "answer": majority_answer, 
                    "score": score,
                    "accuracy": accuracy,
                    'results': results,
                    'ground_truth': answer
                })
        except Exception as e:
            print("Error:", e)
            continue

    print(f"Successfully processed {len(results_all)} questions for {split_name}")

    # Calculate overall accuracy for this split (majority answer vs ground truth)
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
# Load only train split
train_dataset = load_dataset("hiyouga/math12k", split='train')

train_examples = [row for row in train_dataset]

train_questions = [example['problem'] for example in train_examples]
train_answers = [example['answer'] for example in train_examples]

print(f'Loaded {len(train_questions)} train questions from {args.dataset} dataset')

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = vllm.LLM(
    model=args.model,
    tokenizer=args.model,
    tensor_parallel_size=8,
    gpu_memory_utilization=0.85,
    seed=77,
)
sample_params = vllm.SamplingParams(
    max_tokens=8192,
    temperature=0.6,
    top_p=0.95,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=args.num_samples,
)

# Process only train split
train_results, train_accuracy = process_split("train", train_questions, train_answers)

print(f"\n=== Train Results ===")
print(f"Train accuracy: {train_accuracy:.4f} ({len(train_results)} questions)")

# Save score-accuracy analysis data
score_accuracy_data = []
for idx, data in enumerate(train_results):
    score_accuracy_data.append({
        "question_id": idx,
        "question": data['question'],
        "score": data['score'],
        "accuracy": data['accuracy'],
        "ground_truth": data['ground_truth'],
        "majority_answer": data['answer']
    })

score_accuracy_file = f"{args.dataset}_score_accuracy_analysis.json"
with open(score_accuracy_file, "w", encoding="utf-8") as f:
    json.dump(score_accuracy_data, f, ensure_ascii=False, indent=2)
print(f"Score-accuracy analysis data saved to {score_accuracy_file}")

# Create scatter plot for score vs accuracy
scores = [data['score'] for data in train_results]
accuracies = [data['accuracy'] for data in train_results]

if scores and accuracies:
    # Calculate correlation
    correlation, p_value = pearsonr(scores, accuracies)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(scores, accuracies, alpha=0.5, s=50)
    plt.xlabel('Score (Majority Vote Consistency)', fontsize=12)
    plt.ylabel('Accuracy (Correct Rate vs Ground Truth)', fontsize=12)
    plt.title(f'Score vs Accuracy Correlation (Train Set)\nPearson r={correlation:.3f}, p-value={p_value:.4f}\nN={len(scores)} questions', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(scores, accuracies, 1)
    p = np.poly1d(z)
    plt.plot(scores, p(scores), "r--", alpha=0.8, label=f'Trend line: y={z[0]:.2f}x+{z[1]:.2f}')
    plt.legend()
    
    plt.tight_layout()
    scatter_file = f"{args.dataset}_score_accuracy_scatter.png"
    plt.savefig(scatter_file, dpi=300)
    plt.close()
    print(f"Scatter plot saved as {scatter_file}")
    print(f"Correlation coefficient: {correlation:.4f} (p-value: {p_value:.4f})")
    
    if correlation > 0.5:
        print("Strong positive correlation detected between score and accuracy!")
    elif correlation > 0.3:
        print("Moderate positive correlation detected between score and accuracy.")
    elif correlation > 0:
        print("Weak positive correlation detected between score and accuracy.")
    else:
        print("No positive correlation detected between score and accuracy.")

# 构造最终结果
final_results = [{
    "train_accuracy": train_accuracy,
    "train_questions_count": len(train_results),
    "score_accuracy_correlation": correlation if scores and accuracies else None,
    "correlation_p_value": p_value if scores and accuracies else None
}] + train_results

# 本地保存json文件
local_json_path = f"{args.dataset}_majority_results.json"
with open(local_json_path, "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)
print(f"Local results saved to {local_json_path}")

# Direct upload to Hugging Face Hub
print("Preparing data for upload...")
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
train_scores = [data['score'] for data in train_results]

if train_scores:
    print(f"Train score distribution: min={min(train_scores):.3f}, max={max(train_scores):.3f}, mean={sum(train_scores)/len(train_scores):.3f}")

# Create and save score distribution plot
if train_scores:
    plt.figure(figsize=(8, 6))
    plt.hist(train_scores, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Train Score Distribution ({len(train_scores)} questions)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scores_distribution.png', dpi=300)
    plt.close()
    print("Score distribution plot saved as scores_distribution.png")

# Prepare data for upload
train_filtered_data = [
    {
        'problem': data['question'],
        'answer': data['answer'], 
        'score': data['score'],
        'accuracy': data['accuracy']
    } 
    for data in train_results 
    if data['answer'] != '' and data['answer'] != 'None'
]

print(f"Prepared {len(train_filtered_data)} train questions for upload")

if train_filtered_data and HUGGINGFACENAME:
    try:
        repo_name = f"{args.dataset}_consistency"
        
        dataset = Dataset.from_list(train_filtered_data)
        
        print(f"Uploading to {HUGGINGFACENAME}/{repo_name}")
        dataset.push_to_hub(f"{HUGGINGFACENAME}/{repo_name}", private=True)
        print(f"Successfully uploaded to Hugging Face Hub")
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {e}")

print("Processing completed!")