#!/bin/bash

model_name=$1
save_name=$2

pids=()

for i in {0..7}; do
  CUDA_VISIBLE_DEVICES=$i python question_evaluate/evaluate_gpt.py --suffix $i --save_name $save_name &
  pids[$i]=$!
done

wait ${pids[0]}
echo "Task 0 finished."

timeout_duration=36000

(
  sleep $timeout_duration
  echo "Timeout reached. Killing remaining tasks..."
  for i in {1..7}; do
    if kill -0 ${pids[$i]} 2>/dev/null; then
      kill -9 ${pids[$i]} 2>/dev/null
      echo "Killed task $i"
    fi
  done
) &

for i in {1..7}; do
  wait ${pids[$i]} 2>/dev/null
done
