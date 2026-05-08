# Learning to Pose Problems: Reasoning‑Driven and Solver‑Adaptive Data Synthesis

We train a **problem generator** that (i) reasons about *how* to create a new problem before writing it down, and (ii) adapts the difficulty of every new problem to the current solver's ability using the solver's own feedback as a **label-free reward**.

---

## Installation

```bash
conda create -n reasoning-synthesis python=3.10 -y
conda activate reasoning-synthesis
pip install -r requirements.txt
# flash-attn wants the CUDA toolchain at install time:
pip install flash_attn==2.7.4.post1 --no-build-isolation
```

The pinned versions matter: `vllm==0.9.1`, `transformers==4.52.4`,
`torch==2.7.0`, `ray==2.46.0`, `liger_kernel==0.5.10`, `mathruler==0.1.0`.

### Environment variables

```bash
export STORAGE_PATH=/abs/path/for/checkpoints/and/generated_questions
export HUGGINGFACENAME=your-hf-username
export HF_TOKEN=hf_xxxxx
export OPENAI_API_KEY=sk-...                          # for the data-labelling step
# export OPENAI_BASE_URL=https://api.openai.com/v1    # optional, for non-OpenAI providers
# export WANDB_API_KEY=...                            # optional, enables wandb logging
```

### Hardware

The shipped scripts assume an 8-GPU node: GPUs 0–3 run verl's FSDP trainer
and GPUs 4–7 host four vLLM reward servers on ports 5000–5003. For fewer
GPUs, edit `CUDA_VISIBLE_DEVICES` and `tensor_parallel_size` by hand.

---

## Generator training

### Pre-compute `a_ori` on the seed set

`caller_rl.py` needs per-problem solver consistency `a_ori` (= `target_difficulty = 1 - a_ori`) for every seed problem:

```bash
python question_evaluate/evaluate_seed.py \
    --model Qwen/Qwen3-4B-Base \
    --num_samples 10 \
    --dataset math12k
```

Uploads `$HUGGINGFACENAME/math12k_evaluation` to HF Hub.

### Train the generator with solver-feedback RL

```bash
bash scripts/questioner_train_rl.sh \
    Qwen/Qwen3-4B-Base \
    $GEN_SFT \
    qwen3-4b-base_generator_ours
```

The script starts four vLLM reward servers on GPUs 4–7 and runs verl's GRPO
trainer on GPUs 0–3 against `$HUGGINGFACENAME/math12k_evaluation@train`
with the solver-adaptive prompt and `caller_rl.py` reward.

### Synthesise + label + solver RL + evaluate

```bash
bash scripts/solver_train.sh \
    Qwen/Qwen3-4B-Base \
    $GEN_RL \
    Qwen3-4B-Base_solver_ours
```

For the three general-reasoning benchmarks:

```bash
bash evaluation/evaluate_reasoning.bash \
     $STORAGE_PATH/models/Qwen3-4B-Base_solver_ours/global_step_100/actor/huggingface
```

Results are appended to `final_results.jsonl`.

---

## Vision-language track

### One-off data preparation

Re-upload MMK12 with the field names our pipeline expects
(`problem`, `answer`, `images: list`):

```bash
python vldataprocess.py
```

### Pre-compute VL `a_ori`

```bash
python question_evaluate/evaluate_seed_vl.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --num_samples 10 \
    --dataset mmk12 \
    --source_dataset $HUGGINGFACENAME/MMK12
```

Uploads `$HUGGINGFACENAME/mmk12_vl_evaluation` (images are carried through).

### Train the VL generator with solver-feedback RL

```bash
bash scripts/questioner_train_vl.sh \
    Qwen/Qwen2.5-VL-7B-Instruct \
    $GEN_VL_SFT \
    qwen2.5-vl-7b_generator_vl
```

### Synthesise + label + VL solver RL + evaluate

```bash
export GEN_VL=$STORAGE_PATH/models/qwen2.5-vl-7b_generator_vl/global_step_20/actor/huggingface
bash scripts/solver_train_vl.sh \
    Qwen/Qwen2.5-VL-7B-Instruct \
    $GEN_VL \
    Qwen2.5-VL-7B_solver_vl
```
---

## Data artefacts

Local artefacts land in `$STORAGE_PATH` (gitignored):

```
$STORAGE_PATH/
├── generated_question/   raw rollouts and labelled results
├── temp_results/         short-lived reward-function IPC files
├── combined/             combined-dataset stats
├── models/               generator/solver FSDP + HF checkpoints
└── evaluation/<model>/results_<dataset>.json
```

HF Hub uploads under `$HUGGINGFACENAME/`:

```
{dataset}_evaluation     math seeds with `score` = a_ori
{dataset}_vl_evaluation  VL variant with `images` column
{experiment_name}_both   seeds ∪ labelled generated, used for solver GRPO
```

---

## Acknowledgements

- `verl/` is a vendored copy of [EasyR1 / verl](https://github.com/hiyouga/EasyR1)
  (Bytedance, Apache 2.0); our RL training runs inside this trainer.
- The program verifier is [mathruler](https://github.com/hiyouga/MathRuler).
- This project builds on [R-Zero](https://github.com/Chengsong-Huang/R-Zero).

## Citation
If you find Reasoning-Synthesis useful for your research and applications, please cite using this BibTeX:
```bash
@article{wei2025learning,
  title={Learning to pose problems: Reasoning-driven and solver-adaptive data synthesis},
  author={Wei, Yongxian and Zhao, Yilin and Shen, Li and Chen, Xinrui and Cheng, Runxi and Du, Sinan and Yu, Hao and others},
  journal={arXiv preprint arXiv:2511.09907},
  year={2025}
}
