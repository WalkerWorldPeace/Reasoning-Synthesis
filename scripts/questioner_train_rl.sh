#!/bin/bash

set -e

solver_model_path=${1:-Qwen/Qwen3-4B-Base}
questioner_model_path=${2:?"questioner_model_path is required (SFT'd CoT cold-start ckpt)"}
save_path=${3:-qwen3-4b-base_questioner_rl}
dataset=${DATASET:-math12k}
echo "save_path: $save_path"

RUN_ID=$(date +%s%N)
export RUN_ID
echo "RUN_ID=$RUN_ID"

bash vllm_service_init/start_rl.sh $solver_model_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID"

echo "Start training questioner: $questioner_model_path -> $save_path"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$questioner_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=1000 \
    data.train_files=${HUGGINGFACENAME}/${dataset}_evaluation@train \
    data.val_files=${HUGGINGFACENAME}/${dataset}_evaluation@test \
    worker.reward.reward_function=./examples/reward_function/caller_rl.py:compute_score \
    worker.reward.reward_type=batchrl \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=4 \
    data.format_prompt=./examples/format_prompt/questioner_rl.jinja \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=32 \
    data.rollout_batch_size=256 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=32 \
    worker.rollout.max_num_batched_tokens=6144 \
    trainer.max_steps=20 \
    trainer.save_freq=10

sleep 5

echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_20/actor

sleep 10

pkill python

echo "questioner training finished"