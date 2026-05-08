#!/bin/bash

save_name=$1
gpt_model=${2:-gemini-2.5-pro}

pids=()
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python question_evaluate/evaluate_gpt_vl.py \
      --suffix $i --save_name "$save_name" --gpt_model "$gpt_model" &
  pids[$i]=$!
done

wait ${pids[0]}
echo "Task 0 finished."

timeout_duration=36000
(
  sleep $timeout_duration
  echo "Timeout reached. Killing remaining tasks..."
  for i in 1 2 3 4 5 6 7; do
    if kill -0 ${pids[$i]} 2>/dev/null; then
      kill -9 ${pids[$i]} 2>/dev/null
      echo "Killed task $i"
    fi
  done
) &

for i in 1 2 3 4 5 6 7; do
  wait ${pids[$i]} 2>/dev/null
done
