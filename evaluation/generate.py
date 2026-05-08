import vllm
import argparse
import os, sys
import numpy as np
# Add root directory to Python path first
root = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(root)
if root not in sys.path:
    sys.path.append(root)
import  evaluation.datasets_loader as datasets_loader
from transformers import AutoTokenizer
import json

STORAGE_PATH = os.getenv("STORAGE_PATH")
def get_n_samples(dataset_name):
    """根据数据集名称返回对应的n_samples值"""
    if dataset_name in ["aime2024", "aime2025", "amc"]:
        return 4
    else:
        return 1

def main(args):
    print("STORAGE_PATH")
    print(STORAGE_PATH)
    n_samples = get_n_samples(args.dataset)
    print(args.model, args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=0.85,
        # seed=42  # 添加种子确保可重现性
    )
    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        n=n_samples,
        stop_token_ids=[tokenizer.eos_token_id],
        # seed=42  # 添加种子确保可重现性
    )
    handler = datasets_loader.get_dataset_handler(args.dataset,args.name)
    questions, answers = handler.load_data()
    chats=[[{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},{"role": "user", "content": question}] for question in questions]
    if tokenizer.chat_template:
        prompts = [tokenizer.apply_chat_template(chat, tokenize=False,add_generation_prompt=True, add_special_tokens=True) for chat in chats]
    else:
        prompts = ["system: " + chat[0]["content"] + '\n' + "user: " + chat[1]["content"] + '\nPlease reason step by step, and put your final answer within \\boxed{}.' for chat in chats]
    responses = model.generate(prompts, sampling_params=sample_params,use_tqdm=True)

    # 处理多个回复并计算平均准确率
    all_results = []
    question_scores = []
    question_variances = []

    for i, (response, question, answer) in enumerate(zip(responses, questions, answers)):
        response_texts = [output.text for output in response.outputs]
        
        # 为每个回复计算分数
        scores, _ = handler.get_score(response_texts, [answer] * n_samples)
        
        # 计算这个问题的平均分数
        avg_score = sum(scores) / len(scores)
        question_scores.append(avg_score)
        variance = np.var(scores)
        question_variances.append(variance)
        
        # 存储所有结果
        question_result = {
            "question": question,
            "answer": answer,
            "responses": response_texts,
            "individual_scores": scores,
            "average_score": avg_score
        }
        all_results.append(question_result)
    
    # 计算总体平均准确率
    overall_average_score = sum(question_scores) / len(question_scores)
    average_within_question_variance = np.mean(question_variances)
    print(f"Overall average accuracy: {overall_average_score:.4f}")

    results = {
        "overall_average_score": overall_average_score,
        "average_within_question_variance": float(average_within_question_variance),
        "config": {
            "n_samples": n_samples,
            "model": args.model
        },
        "all_results": all_results
    }
    
    os.makedirs(f"{STORAGE_PATH}/evaluation/{args.model.replace('/', '_')}", exist_ok=True)
    with open(f"{STORAGE_PATH}/evaluation/{args.model.replace('/', '_')}/results_{args.dataset}.json", "w") as f:
        json.dump(results, f, indent=4)
    # responses = [response.outputs[0].text for response in responses]
    # scores,average_score = handler.get_score(responses, answers)
    # results = [{"question": question, "answer": answer, "response": response, "score": score} for question, answer, response, score in zip(questions, answers, responses, scores)]
    # print(f"Average score: {average_score}")
    # results.append({"average_score": average_score})
    # os.makedirs(f"{STORAGE_PATH}/evaluation/{args.model.replace('/', '_')}", exist_ok=True)
    # with open(f"{STORAGE_PATH}/evaluation/{args.model.replace('/', '_')}/results_{args.dataset}.json", "w") as f:
    #     json.dump(results, f, indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", type=str, default="math")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    main(args)