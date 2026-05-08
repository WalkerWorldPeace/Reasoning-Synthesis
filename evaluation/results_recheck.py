import json
from mathruler.grader import extract_boxed_content, grade_answer
from tqdm import tqdm
import argparse
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
args = parser.parse_args()

STORAGE_PATH = os.getenv("STORAGE_PATH")
# An optional GPT-4o "rechecker" fallback for responses whose exact-match score
# is < 0.5. Configure via standard OpenAI-compatible env vars:
#   OPENAI_API_KEY    required to enable the rechecker
#   OPENAI_BASE_URL   optional (e.g. OpenRouter / Gemini OpenAI-compatible endpoint)
# If OPENAI_API_KEY is not set, the rechecker is inert (returns "No") and the
# script still runs — it just does not "rescue" low-score responses.
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # None => OpenAI default

# 添加线程锁
file_lock = threading.Lock()

def process_example(answer, response):
    if not _OPENAI_API_KEY:
        # Rechecker disabled — behave as if the rechecker marked the sample wrong.
        return "No"
    try:
        client = OpenAI(api_key=_OPENAI_API_KEY, base_url=_OPENAI_BASE_URL)
        
        # 调用API
        response_obj = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a math answer checker."},
                {"role": "user", "content": f"Hi, there is a answer: {answer}\n\n, and the ground truth answer is: {response}\n\n, please check whether the answer is correct or not, and return the **only** Yes or No."}
            ],
            temperature=0.1,
            timeout=20
        )
        
        # 获取响应内容
        result = response_obj.choices[0].message.content
        print(f"API Response: {result}")
        return result
    except Exception as e:
        print('error', e)
        return "No"

def process_dataset(model_name, dataset):
    """处理单个数据集的函数"""
    try:
        print(f"开始处理数据集: {dataset}")
        
        with open(f'{STORAGE_PATH}/evaluation/{model_name.replace("/","_")}/results_{dataset}.json', 'r') as f:
            results = json.load(f)["all_results"]

        for i in tqdm(range(len(results)), desc=f"Processing {dataset}"):
            if 'responses' in results[i] and isinstance(results[i]['responses'], list):
                # 多个回复的情况
                correct_count = 0
                for j, response in enumerate(results[i]['responses']):
                    if results[i]['individual_scores'][j] < 0.5:
                        # gpt_check = process_example(results[i]['answer'], response)
                        # if "yes" in gpt_check.lower():
                        #     correct_count += 1
                        continue
                    else:
                        correct_count += 1
                results[i]['score'] = correct_count / len(results[i]['responses'])
            else:
                # 单个回复的情况，保持原逻辑
                gpt_check = process_example(results[i]['answer'], results[i]['response'])
                if "yes" in gpt_check.lower():
                    results[i]['score'] = 1
        
        # 计算数据集得分
        dataset_score = round(sum([result['score'] for result in results])/len(results)*100, 2)
        
        dataset_result = {
            'model': model_name,
            'dataset': dataset,
            'score': dataset_score
        }
        
        # 使用线程锁保护文件写入
        with file_lock:
            with open(f'final_results.jsonl', 'a') as f:
                json.dump(dataset_result, f)
                f.write('\n')
        
        print(f"数据集 {dataset} 处理完成，得分: {dataset_score}")
        return dataset_result
        
    except Exception as e:
        print(f"处理数据集 {dataset} 时出错: {e}")
        return {
            'model': model_name,
            'dataset': dataset,
            'score': 0,
            'error': str(e)
        }

# 主处理逻辑
new_results = []
datasets = [
    "math",
    "gsm8k", 
    "amc",
    "minerva",
    "olympiad",
    "aime2024",
    "aime2025",
]

for model_name in [args.model_name]:
    print(f"开始处理模型: {model_name}")
    
    # 使用线程池并行处理数据集
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交所有任务
        future_to_dataset = {
            executor.submit(process_dataset, model_name, dataset): dataset 
            for dataset in datasets
        }
        
        # 收集结果
        for future in as_completed(future_to_dataset):
            dataset = future_to_dataset[future]
            try:
                result = future.result()
                new_results.append(result)
                print(f"完成: {result}")
            except Exception as exc:
                print(f"数据集 {dataset} 产生异常: {exc}")
                new_results.append({
                    'model': model_name,
                    'dataset': dataset,
                    'score': 0,
                    'error': str(exc)
                })

print("所有数据集处理完成!")
print("最终结果:")
for result in new_results:
    print(f"{result['model']} - {result['dataset']}: {result['score']}%")




