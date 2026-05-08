model_name=$1
num_samples=$2
save_name=$3
export VLLM_DISABLE_COMPILE_CACHE=1
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python question_generate/question_generate_vl.py \
    --model $model_name --suffix $i --num_samples $num_samples --save_name $save_name &
done
wait
