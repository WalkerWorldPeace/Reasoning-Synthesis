import json
import os
import argparse
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login
import random

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
    
    print(f"Loading original math12k dataset...")
    
    # 1. 加载原始 math12k 数据集
    original_dataset = load_dataset("hiyouga/math12k", split="train")
    original_questions = [item["problem"] for item in original_dataset]
    original_answers = [item["answer"] for item in original_dataset]
    
    # 2. 重现种子采样逻辑，收集所有被选中的原始题目
    selected_originals = []
    for seed in range(8):
        title_random = random.Random(seed)
        if args.num_samples_per_seed > len(original_questions):
            selected_indices = title_random.choices(range(len(original_questions)), k=args.num_samples_per_seed)
        else:
            selected_indices = title_random.sample(range(len(original_questions)), args.num_samples_per_seed)
        
        for idx in selected_indices:
            selected_originals.append({
                'problem': original_questions[idx],
                'answer': original_answers[idx],
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
                        
                        generated_problems.append({
                            'problem': item['question'],
                            'answer': item['answer'],
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
    elif args.combine_strategy == 'custom':
        # ✅ 新增：自定义生成题采样数量
        target_generated_count = args.generated_sample_count
        if len(generated_problems) > target_generated_count:
            sampled_generated = random.sample(generated_problems, target_generated_count)
        else:
            sampled_generated = generated_problems
            print(f"Warning: Only {len(generated_problems)} generated problems available, "
                  f"less than target {target_generated_count}")
        train_data = selected_originals + sampled_generated
    else:
        raise ValueError(f"Unknown combine_strategy: {args.combine_strategy}")
    
    # 5. 打乱数据顺序
    random.shuffle(train_data)
    
    # 6. 格式化为训练所需的格式
    formatted_data = []
    for item in train_data:
        formatted_data.append({
            'problem': item['problem'],
            'answer': item['answer']
        })
    
    print(f"Final training dataset size: {len(formatted_data)}")
    print(f"  - Original problems: {len([x for x in train_data if x['source'] == 'original'])}")
    print(f"  - Generated problems: {len([x for x in train_data if x['source'] == 'generated'])}")
    
    # 7. 创建数据集并上传
    train_dataset = Dataset.from_list(formatted_data)
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
    
    with open(f'{STORAGE_PATH}/combined/dataset_stats_{repo_name}.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Dataset statistics saved to {STORAGE_PATH}/combined/dataset_stats_{repo_name}.json")
    print(f"Dataset uploaded successfully: {HUGGINGFACENAME}/{repo_name}")
    
    return repo_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine original and generated problems for training")
    parser.add_argument("--experiment_name", type=str, default="Qwen3-4B-Base_solver_nosft_math12k_newrl_math12k",
                       help="Experiment name to find generated question files")
    parser.add_argument("--num_samples_per_seed", type=int, default=1000,
                       help="Number of samples generated per seed")
    parser.add_argument("--min_score", type=float, default=0.0,
                       help="Minimum score for generated problems")
    parser.add_argument("--max_score", type=float, default=1.0,
                       help="Maximum score for generated problems")
    parser.add_argument("--combine_strategy", type=str, 
                       choices=['original_only', 'generated_only', 'both', 'balanced', 'custom'],  # ✅ 添加新选项
                       default='both',
                       help="Strategy for combining datasets")
    parser.add_argument("--generated_sample_count", type=int, default=2000,  # ✅ 新增参数
                       help="Number of generated problems to sample (only for custom_ratio strategy)")
    
    args = parser.parse_args()
    main(args)