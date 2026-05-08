#!/bin/bash

set -e

solver_model_path=${1:-Qwen/Qwen2.5-VL-7B-Instruct}
questioner_model_path=${2:?"questioner_model_path is required (RL-trained VL generator ckpt)"}
experiment_name=${3:-Qwen2.5-VL-7B_solver_vl}

echo "STORAGE_PATH=$STORAGE_PATH"
echo "start train VL solver $experiment_name $solver_model_path $questioner_model_path"

export VLLM_DISABLE_COMPILE_CACHE=1
echo 'start generate question'
bash question_generate/question_generate_vl.bash ${questioner_model_path} 500 ${experiment_name}

echo 'start label generated questions'
bash question_evaluate/evaluate_gpt_vl.sh ${experiment_name}

echo 'start combine datasets and upload to HuggingFace'
python combine_datasets_vl.py \
    --experiment_name ${experiment_name} \
    --max_score 1 --min_score 0.4 \
    --combine_strategy both \
    --num_samples_per_seed 500

pkill python

echo 'start train solver'
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    data.train_files=${HUGGINGFACENAME}/${experiment_name}_both@train \
    data.val_files=${HUGGINGFACENAME}/MMK12@test \
    worker.actor.model.model_path=$solver_model_path \
    algorithm.kl_coef=1e-2 \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    trainer.total_epochs=100 \
    trainer.max_steps=100 \
    trainer.val_freq=10 \
    trainer.val_before_train=False \
    trainer.save_freq=50 \
    trainer.save_limit=-1 \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=16

echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_100/actor
