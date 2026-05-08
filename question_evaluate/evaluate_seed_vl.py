import json
import vllm
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import re
import os, sys
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Image as HFImage
from huggingface_hub import login
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Add root directory to Python path first
root = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(root)
if root not in sys.path:
    sys.path.append(root)
from evaluation.datasets_loader import get_dataset_handler
from mathruler.grader import extract_boxed_content, grade_answer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--num_samples", type=int, default=10)
parser.add_argument("--dataset", type=str, default="mmk12",
                    help="Label suffix for the uploaded dataset "
                         "(pushed as $HUGGINGFACENAME/{dataset}_vl_evaluation).")
parser.add_argument("--source_dataset", type=str, default=None,
                    help="HF dataset id to evaluate. Defaults to $HUGGINGFACENAME/MMK12.")
parser.add_argument("--tensor_parallel_size", type=int, default=4,
                    help="vLLM tensor_parallel_size for the solver.")
args = parser.parse_args()
STORAGE_PATH = os.getenv("STORAGE_PATH")
if STORAGE_PATH is None:
    raise RuntimeError("STORAGE_PATH env var is not set")
HUGGINGFACENAME = os.getenv("HUGGINGFACENAME")
if HUGGINGFACENAME is None:
    raise RuntimeError("HUGGINGFACENAME env var is not set")
SOURCE_DATASET = args.source_dataset or f"{HUGGINGFACENAME}/MMK12"

def extract_answer(response):
    match = re.search(r"\\boxed{(.*?)}", response)
    return match.group(1) if match else None

def process_split(split_name, questions, answers, images_list):
    """Run multimodal majority-vote evaluation on one split."""
    print(f"Processing {split_name} split with {len(questions)} questions...")

    prompts_with_images = []
    min_size = 28  # Qwen2.5-VL minimum side length

    for question, images in zip(questions, images_list):
        processed_images = []
        if images:
            for img in images:
                try:
                    if isinstance(img, dict):
                        pil_img = Image.open(BytesIO(img["bytes"]))
                    elif isinstance(img, Image.Image):
                        pil_img = img
                    else:
                        continue

                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')

                    width, height = pil_img.size
                    if width < min_size or height < min_size:
                        scale = max(min_size / width, min_size / height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
                        print(f"Warning: Resized small image from {width}x{height} to {new_width}x{new_height}")

                    processed_images.append(pil_img)
                except Exception as e:
                    print(f"Error processing image: {e}")
                    continue

        if processed_images:
            content_list = [{"type": "image"} for _ in processed_images]
            content_list.append({"type": "text", "text": question})

            chat = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": content_list},
            ]

            prompt = processor.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=True,
            )

            prompts_with_images.append({
                "prompt": prompt,
                "multi_modal_data": {"image": processed_images},
            })
        else:
            # No valid images -> text-only prompt
            chat = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": question},
            ]

            prompt = processor.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=True,
            )

            prompts_with_images.append({"prompt": prompt})

    print(f"Generating responses for {split_name}...")
    responses = model.generate(prompts_with_images, sampling_params=sample_params, use_tqdm=True)

    print(f"Processing {len(responses)} responses for {split_name}...")
    results_all = []

    for response, answer, question, images in zip(responses, answers, questions, images_list):
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

            if error_flag or not answer_counts:
                continue

            max_count = max(answer_counts.values())
            majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
            score = max_count / len(results)

            if "证明" in question or 'box' in question.lower() or 'text' in majority_answer.lower():
                continue

            # Extract first image (if any) with the same RGB / min-size normalisation.
            first_image = None
            if images:
                img = images[0]
                try:
                    if isinstance(img, dict):
                        first_image = Image.open(BytesIO(img["bytes"]))
                    elif isinstance(img, Image.Image):
                        first_image = img

                    if first_image and first_image.mode != 'RGB':
                        first_image = first_image.convert('RGB')

                    if first_image:
                        width, height = first_image.size
                        if width < min_size or height < min_size:
                            scale = max(min_size / width, min_size / height)
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            first_image = first_image.resize((new_width, new_height), Image.LANCZOS)
                except Exception as e:
                    print(f"Error loading result image: {e}")
                    first_image = None

            results_all.append({
                "question": question,
                "answer": majority_answer,
                "score": score,
                'results': results,
                'ground_truth': answer,
                'image': first_image,
            })
        except Exception as e:
            print(f"Error processing response: {e}")
            continue

    print(f"Successfully processed {len(results_all)} questions for {split_name}")

    # Calculate accuracy
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

print(f'Loading {SOURCE_DATASET}...')
train_dataset = load_dataset(SOURCE_DATASET, split='train')
test_dataset = load_dataset(SOURCE_DATASET, split='test')

train_examples = [row for row in train_dataset]
train_questions = [example['problem'] for example in train_examples]
train_answers = [example['answer'] for example in train_examples]
train_images = [example['images'] for example in train_examples]

test_examples = [row for row in test_dataset]
test_questions = [example['problem'] for example in test_examples]
test_answers = [example['answer'] for example in test_examples]
test_images = [example['images'] for example in test_examples]
print(f'Loaded {len(train_questions)} train questions and {len(test_questions)} test questions from {SOURCE_DATASET}')


# Initialize model, tokenizer and processor
print("Initializing model and processor...")
tokenizer = AutoTokenizer.from_pretrained(args.model)
processor = AutoProcessor.from_pretrained(args.model)
model = vllm.LLM(
    model=args.model,
    tensor_parallel_size=args.tensor_parallel_size,
    gpu_memory_utilization=0.85,
    seed=77,
)
sample_params = vllm.SamplingParams(
    max_tokens=4096,
    temperature=0.6,
    top_p=0.95,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=args.num_samples,
)

# Process train split
test_results, test_accuracy = process_split("test", test_questions, test_answers, test_images)
train_results, train_accuracy = process_split("train", train_questions, train_answers, train_images)
overall_accuracy = (train_accuracy * len(train_results) + test_accuracy * len(test_results)) / (len(train_results) + len(test_results)) if (len(train_results) + len(test_results)) > 0 else 0.0
print(f"\n=== Results ===")
print(f"Train accuracy: {train_accuracy:.4f} ({len(train_results)} questions)")
print(f"Test accuracy: {test_accuracy:.4f} ({len(test_results)} questions)")
print(f"Overall accuracy: {overall_accuracy:.4f}")


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
all_results = train_results + test_results
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
        'score': data['score'],
        'images': [data['image']] if data.get('image') else []
    } 
    for data in train_results 
    if data['answer'] != '' and data['answer'] != 'None'
]

test_filtered_data = [
    {
        'problem': data['question'],
        'answer': data['answer'], 
        'score': data['score'],
        'images': [data['image']] if data.get('image') else []
    } 
    for data in test_results 
    if data['answer'] != '' and data['answer'] != 'None'
]

print(f"Prepared {len(train_filtered_data)} train questions and {len(test_filtered_data)} test questions for upload")

# Upload to Hugging Face Hub if we have data
if (train_filtered_data or test_filtered_data) and HUGGINGFACENAME:
    try:
        repo_name = f"{args.dataset}_vl_evaluation"
        
        features = Features({
            'problem': Value('string'),
            'answer': Value('string'),
            'score': Value('float'),
            'images': [HFImage()]
        })
        
        dataset_dict = {}
        if train_filtered_data:
            dataset_dict["train"] = Dataset.from_list(train_filtered_data, features=features)
        if test_filtered_data:
            dataset_dict["test"] = Dataset.from_list(test_filtered_data, features=features)
        
        dataset = DatasetDict(dataset_dict)
        
        print(f"Uploading to {HUGGINGFACENAME}/{repo_name}")
        dataset.push_to_hub(f"{HUGGINGFACENAME}/{repo_name}", private=True)
        print(f"Successfully uploaded {len(train_filtered_data)} train and {len(test_filtered_data)} test questions to Hugging Face Hub")
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {e}")
        import traceback
        traceback.print_exc()
else:
    if not (train_filtered_data or test_filtered_data):
        print("No data to upload after filtering")
    if not HUGGINGFACENAME:
        print("HUGGINGFACENAME environment variable not set, skipping upload")

print("Processing completed!")