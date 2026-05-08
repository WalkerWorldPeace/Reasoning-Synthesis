import json
import os
import argparse
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Image as HFImage
from huggingface_hub import login
import random
import base64
from io import BytesIO
from PIL import Image

def base64_to_image(base64_str):
    """将base64字符串转换回PIL Image""" 
    img_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_data))

def _login_hf():
    """Login to Hugging Face using HF_TOKEN env var, falling back to tokens.json."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token is None and os.path.exists("tokens.json"):
        with open("tokens.json", "r") as f:
            token = json.load(f).get("huggingface")
    if token is None:
        raise RuntimeError(
            "No Hugging Face token found. Set HF_TOKEN env var or "
            "create tokens.json from tokens.json.example."
        )
    login(token=token)


def main(args):
    _login_hf()

    STORAGE_PATH = os.getenv("STORAGE_PATH")
    HUGGINGFACENAME = os.getenv("HUGGINGFACENAME")
    if STORAGE_PATH is None or HUGGINGFACENAME is None:
        raise RuntimeError("Please set STORAGE_PATH and HUGGINGFACENAME env vars")

    seed_dataset = args.seed_dataset or f"{HUGGINGFACENAME}/MMK12"
    print(f"Loading original VL seed dataset: {seed_dataset}...")

    # 1. 加载原始 VL seed 数据集（默认 MMK12）
    original_dataset = load_dataset(seed_dataset, split="train")
    original_questions = [item["problem"] for item in original_dataset]
    original_answers = [item["answer"] for item in original_dataset]
    original_images = [item["images"] for item in original_dataset]
    
    # 2. 重现种子采样逻辑，收集所有被选中的原始题目
    selected_originals = []
    for seed in range(8):
        title_random = random.Random(seed)
        if args.num_samples_per_seed > len(original_questions):
            selected_indices = title_random.choices(range(len(original_questions)), k=args.num_samples_per_seed)
        else:
            selected_indices = title_random.sample(range(len(original_questions)), args.num_samples_per_seed)
        
        for idx in selected_indices:
            # 获取第一张图片
            images = original_images[idx]
            first_image = None
            if images and len(images) > 0:
                img = images[0]
                if isinstance(img, dict):
                    first_image = Image.open(BytesIO(img["bytes"]))
                elif isinstance(img, Image.Image):
                    first_image = img
            
            selected_originals.append({
                'problem': original_questions[idx],
                'answer': original_answers[idx],
                'image': first_image,  # PIL Image对象
                'source': 'original',
                'seed': seed
            })
    
    print(f"Collected {len(selected_originals)} original problems from seeds 0-7")
    
    # 3. 加载生成的题目
    generated_problems = []
    for seed in range(8):
        try:
            file_path = f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{seed}_results.json'
            with open(file_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    if (item['score'] >= args.min_score and 
                        item['score'] <= args.max_score and 
                        item['answer'] != '' and 
                        item['answer'] != 'None' and
                        item['question'].strip() != ''):
                        
                        # 将base64转换为PIL Image
                        pil_image = None
                        if item.get('image'):
                            try:
                                pil_image = base64_to_image(item['image'])
                            except Exception as e:
                                print(f"Warning: Failed to decode image for seed {seed}: {e}")
                                continue
                        
                        generated_problems.append({
                            'problem': item['question'],
                            'answer': item['answer'],
                            'image': pil_image,  # PIL Image对象
                            'source': 'generated',
                            'seed': seed,
                            'score': item['score']
                        })
        except FileNotFoundError:
            print(f"Warning: File {args.experiment_name}_{seed}_results.json not found")
            continue
    
    print(f"Collected {len(generated_problems)} generated problems")
    
    # 4. 根据组合策略创建训练数据集
    if args.combine_strategy == 'original_only':
        train_data = selected_originals
    elif args.combine_strategy == 'generated_only':
        train_data = generated_problems
    elif args.combine_strategy == 'both':
        train_data = selected_originals + generated_problems
    elif args.combine_strategy == 'balanced':
        # 平衡数量：从生成的题目中采样与原始题目相同的数量
        if len(generated_problems) > len(selected_originals):
            sampled_generated = random.sample(generated_problems, len(selected_originals))
        else:
            sampled_generated = generated_problems
        train_data = selected_originals + sampled_generated
    else:
        raise ValueError(f"Unknown combine_strategy: {args.combine_strategy}")
    
    # 5. 打乱数据顺序
    random.shuffle(train_data)
    
    # 6. 格式化为训练所需的格式(包含图片)
    formatted_data = []
    for item in train_data:
        # 处理问题文本，确保<image>标记数量正确
        problem = item['problem']
        
        # 移除问题中所有的<image>标记
        problem_clean = problem.replace('<image>', '').strip()
        
        # 根据实际图片数量添加<image>标记
        if item['image'] is not None:
            # 只有一张图片，添加一个<image>标记
            problem_with_image = f"<image>{problem_clean}"
        else:
            # 没有图片
            problem_with_image = problem_clean
        
        formatted_data.append({
            'problem': problem_with_image,
            'answer': item['answer'],
            'images': [item['image']] if item['image'] else []  # 以列表形式保存图片
        })
    
    print(f"Final training dataset size: {len(formatted_data)}")
    print(f"  - Original problems: {len([x for x in train_data if x['source'] == 'original'])}")
    print(f"  - Generated problems: {len([x for x in train_data if x['source'] == 'generated'])}")
    
    # 统计有图片和无图片的样本数量
    with_image_count = len([x for x in formatted_data if x['images']])
    without_image_count = len([x for x in formatted_data if not x['images']])
    print(f"  - With images: {with_image_count}")
    print(f"  - Without images: {without_image_count}")
    
    # 7. 创建数据集并上传 (使用HuggingFace的Image特性)
    # 定义数据集特性，包括图片类型
    features = Features({
        'problem': Value('string'),
        'answer': Value('string'),
        'images': [HFImage()]  # 图片列表
    })
    
    train_dataset = Dataset.from_list(formatted_data, features=features)
    dataset_dict = DatasetDict({"train": train_dataset})
    
    # 构建仓库名称
    repo_name = f"{args.experiment_name}_{args.combine_strategy}"
    
    print(f"Uploading to {HUGGINGFACENAME}/{repo_name}...")
    dataset_dict.push_to_hub(f"{HUGGINGFACENAME}/{repo_name}", private=True)
    
    # 8. 保存统计信息
    stats = {
        'total_samples': len(formatted_data),
        'original_samples': len([x for x in train_data if x['source'] == 'original']),
        'generated_samples': len([x for x in train_data if x['source'] == 'generated']),
        'combine_strategy': args.combine_strategy,
        'min_score': args.min_score,
        'max_score': args.max_score,
        'repo_name': repo_name
    }
    
    if len(generated_problems) > 0:
        scores = [x['score'] for x in generated_problems]
        stats['generated_score_stats'] = {
            'mean': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores),
            'count_by_seed': {seed: len([x for x in generated_problems if x['seed'] == seed]) 
                             for seed in range(8)}
        }
    
    os.makedirs(f'{STORAGE_PATH}/combined', exist_ok=True)
    with open(f'{STORAGE_PATH}/combined/dataset_stats_{repo_name}.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Dataset statistics saved to {STORAGE_PATH}/combined/dataset_stats_{repo_name}.json")
    print(f"Dataset uploaded successfully: {HUGGINGFACENAME}/{repo_name}")
    
    return repo_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine original and generated vision-language problems for training")
    parser.add_argument("--experiment_name", type=str, required=True,
                       help="Experiment name to find generated question files")
    parser.add_argument("--seed_dataset", type=str, default=None,
                       help="HF id of the VL seed corpus to sample original problems from. "
                            "Defaults to $HUGGINGFACENAME/MMK12.")
    parser.add_argument("--num_samples_per_seed", type=int, default=1000,
                       help="Number of samples generated per seed")
    parser.add_argument("--min_score", type=float, default=0.0,
                       help="Minimum score for generated problems")
    parser.add_argument("--max_score", type=float, default=1.0,
                       help="Maximum score for generated problems")
    parser.add_argument("--combine_strategy", type=str,
                       choices=['original_only', 'generated_only', 'both', 'balanced'],
                       default='both',
                       help="Strategy for combining datasets")

    args = parser.parse_args()
    main(args)