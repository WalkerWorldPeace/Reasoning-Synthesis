#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Vision-Language Version: This script supports multimodal inputs (text + images) using Qwen2.5-VL.
It employs the 'stopit' library for thread-safe timeout control and optimizes comparison logic.

Setup Instructions:
    # 1. Install required libraries
    pip install stopit qwen-vl-utils

    # 2. Run the server
    python start_vllm_server_rl_vl.py --port 5000 --model_path Qwen/Qwen2.5-VL-7B-Instruct
'''

from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os, sys
import threading
import time
import torch
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from io import BytesIO
import base64

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from mathruler.grader import extract_boxed_content, grade_answer
import stopit

# ------------------------- Command-Line Arguments ------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
parser.add_argument('--gpu_mem_util', type=float, default=0.9)
args = parser.parse_args()

# ------------------------- Helper Functions ------------------------- #
def base64_to_image(base64_str):
    """将base64字符串转换为PIL Image"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_data))

# ------------------------- vLLM Initialization ------------------------ #
print('[init] Loading vision-language model...')

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util
)

sample_params = vllm.SamplingParams(
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=10,
)

# ---------------------- GPU Idle Utilization Thread ---------------------- #
stop_event = threading.Event()
pause_event = threading.Event()

def gpu_idle_worker():
    '''GPU空闲时保持占用，防止性能下降'''
    print('[idle_worker] GPU idle worker started.')
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                print('[idle_worker] Paused.')
                running = False
            time.sleep(0.1)
            continue
        else:
            if not running:
                print('[idle_worker] Resumed.')
                running = True
        try:
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'[idle_worker] Caught a RuntimeError: {e}. Sleeping for 1s...')
            time.sleep(1)
    print('[idle_worker] GPU idle worker stopped.')

idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
idle_thread.start()

# ------------------------ Timeout Utility --------------------------- #
@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """带超时的答案评分函数"""
    return grade_answer(res1, res2)

# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    '''多模态处理端点: 读取任务文件，调用vLLM，整合答案，写入结果'''
    
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[server] Received request for task file: {name}')

    # ---------- Load Data ----------
    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    questions = [item.get('question', '') for item in data]
    answers = [item.get('answer', '') for item in data]
    images = [item.get('image', None) for item in data]  # Base64 字符串或 None
    original_index = [item.get('original_index', '') for item in data]

    # ---------- Prepare Multimodal Inputs ----------
    valid_indices, valid_questions, valid_answers, prompts_with_images = [], [], [], []
    
    for i, (q, a, img_data) in enumerate(zip(questions, answers, images)):
        if q:
            valid_indices.append(i)
            valid_questions.append(q)
            valid_answers.append(a)
            
            # 处理图片 - img_data 是 base64 字符串
            processed_images = []
            if img_data:
                try:
                    pil_img = base64_to_image(img_data)
                    processed_images.append(pil_img)
                except Exception as e:
                    print(f'[server] Error processing image for question {i}: {e}')
            
            # 构建多模态消息
            content_list = []
            for _ in processed_images:
                content_list.append({"type": "image"})
            
            content_list.append({
                "type": "text",
                "text": q
            })
            
            chat = [{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},{"role": "user", "content": content_list}]
        
            
            # 应用chat template
            prompt = processor.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=True
            )
            
            prompts_with_images.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": processed_images
                }
            })
    
    print(f'[server] Prepared {len(prompts_with_images)} valid multimodal prompts.')

    # ---------- vLLM Generation ----------
    if prompts_with_images:
        responses = model.generate(prompts_with_images, sampling_params=sample_params, use_tqdm=True)
    else:
        responses = []
    print('[server] Generation completed.')

    # ---------- Results Post-Processing ----------
    def process_single(question, golden_answer, response):
        '''整合并评分单个问题的vLLM输出'''
        results = [extract_boxed_content(out.text) for out in response.outputs]
        results = [res for res in results if res and res.strip() and res.strip().lower() != 'none']

        answer_counts = {}
        for res in results:
            if not res:
                continue
            matched = False
            
            for exist_ans in list(answer_counts.keys()):
                # 先进行简单比较
                if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                    answer_counts[exist_ans] += 1
                    matched = True
                    break
                
                # 复杂比较带超时
                try:
                    is_match = False
                    match_result_1 = grade_answer_with_timeout(res, exist_ans, timeout=10)
                    if match_result_1 == 'TIMED_OUT':
                        print(f"      [grader] TIMEOUT comparing '{res[:30]}...' with '{exist_ans[:30]}...'.")
                    elif match_result_1:
                        is_match = True

                    if not is_match:
                        match_result_2 = grade_answer_with_timeout(exist_ans, res, timeout=10)
                        if match_result_2 == 'TIMED_OUT':
                            print(f"      [grader] TIMEOUT comparing '{exist_ans[:30]}...' with '{res[:30]}...'. Skipping pair.")
                        elif match_result_2:
                            is_match = True
                    
                    if is_match:
                        answer_counts[exist_ans] += 1
                        matched = True
                        break

                except Exception as e:
                    print(f"      [grader] ERROR comparing '{res[:30]}...' with '{exist_ans[:30]}...': {e}. Skipping.")
                    continue
            
            if not matched:
                answer_counts[res] = 1

        if not answer_counts:
            majority_ans, max_count = '', 0
        else:
            majority_ans = max(answer_counts, key=answer_counts.get)
            max_count = answer_counts[majority_ans]

        score = max_count / len(results) if results else 0.0
        return {
            'question': question,
            'answer': majority_ans,
            'score': score if score > 0.1 else 0,
            'results': results
        }

    results_all = []
    response_idx = 0
    for q, a, idx in zip(questions, answers, original_index):
        try:
            if q:
                response = responses[response_idx]
                response_idx += 1
                item = process_single(q, a, response)
                item['original_index'] = idx
                # 不保存 image 字段
                results_all.append(item)
            else:
                results_all.append({
                    'question': q, 
                    'answer': a, 
                    'score': -1, 
                    'results': [], 
                    'original_index': idx
                })
        except Exception as e:
            print(f'[server] CRITICAL: Error processing question: {q}')
            print(f'[server] Error details: {e}')
            results_all.append({
                'question': q,
                'answer': a,
                'score': -1,
                'results': [],
                'error': f'unhandled exception: {str(e)}',
                'original_index': idx
            })
    
    print('[server] All results have been processed.')

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    pause_event.clear()
    print(f'[server] Processed {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})

# ------------------------- Main Application Entrypoint --------------------------- #
if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)
    finally:
        stop_event.set()
        idle_thread.join()
        print('[main] Application shutdown complete.')