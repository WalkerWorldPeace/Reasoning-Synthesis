#!/bin/bash
export VLLM_DISABLE_COMPILE_CACHE=1
model_name=$1

python evaluation/eval_supergpqa.py --model_path $model_name
python evaluation/eval_bbeh.py --model_path $model_name
python evaluation/eval_mmlupro.py --model_path $model_name

echo "==> All tasks have finished!"