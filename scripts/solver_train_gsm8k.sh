#!/bin/bash

set -e

solver_model_path=${1:-Qwen/Qwen3-4B-Base}
questioner_model_path=${2:?"questioner_model_path is required"}
experiment_name=${3:-Qwen3-4B-Base_solver_gsm8k_seed}

echo "STORAGE_PATH=$STORAGE_PATH"

echo "start train solver $experiment_name $solver_model_path $questioner_model_path"

export VLLM_DISABLE_COMPILE_CACHE=1
echo 'start upload'
python combine_datasets.py --experiment_name ${experiment_name} --max_score 1 --min_score 0.4 --combine_strategy both --num_samples_per_seed 500

pkill python

echo 'start train'
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    data.train_files=${HUGGINGFACENAME}/${experiment_name}_both@train \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    worker.actor.model.model_path=$solver_model_path \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    trainer.total_epochs=100 \
    trainer.max_steps=100 \
    trainer.val_freq=10 \
    trainer.val_before_train=False \
    trainer.save_freq=50 \
    trainer.save_limit=-1 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8

echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_100/actor

sleep 10

echo "solver training finished"
bash evaluation/evaluate.bash ${STORAGE_PATH}/models/${experiment_name}/global_step_100/actor/huggingface