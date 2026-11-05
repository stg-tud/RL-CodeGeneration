
# RL-CodeGeneration

Official implementation of the paper:

**"Domain-Adaptable Reinforcement Learning for Code Generation with Dense Rewards"**

This repository provides a unified reinforcement learning framework for improving large language models (LLMs) on code generation tasks using:

- Proximal Policy Optimization (PPO)
- Guided Generation (SynCode-based syntax reward)
- Static analysis (Ruff linter)
- Execution-based rewards (Pass@1, RoboSim)
- KL-regularized policy optimization
- Parameter-efficient fine-tuning via LoRA

The framework supports:

- General-purpose Python generation (OpenCodeInstruct → MBPP / EvalPlus)
- Robotics program synthesis (Robo-Instruct → RoboEval)

---
# Overview

This framework enables **multi-component reward shaping** for LLM fine-tuning and introduces:

- **Dense token-level reward redistribution**  
  (see `rewards/reward_helper.py`)

- **Syntax-aware learning without hard constrained decoding**  
  (see `wrappers/syncode_wrapper.py`)

- **Simulation-based robotics feedback**  
  (see `rewards/robo_instruct_sim_reward_helper.py`)

- **Task-agnostic PPO-based fine-tuning**  
  (see `wrappers/ppo_wrapper.py`)

- **Modular reward engineering**  
  (see `rewards/extra_rewards.py`)

The design is extensible and allows systematic experimentation with reward functions and RL configurations.

---
# Installation

Create and activate the conda environment:

```bash
conda env create -f config/environment.yml
conda activate code_gen
```
Install syncode separately

```bash
pip install --no-deps syncode==0.4.16
```

If using DeepSpeed, ensure compatibility with your CUDA and PyTorch versions.

---

# Configuration

All RL hyperparameters are defined in:

```
config/hyperparams.json
```

Each top-level block (e.g. `ppo`, `ppo_code_gen`) bundles a dataset, a model and a PPO configuration, and is selected with `--param <block>`. The reward weighting is selected separately with `--framework_params` (`code_gen` or `robo`), from the file's `framework` block. SFT has its own file, `sft/sft_hyperparams.json` (see [Supervised Fine-Tuning](#supervised-fine-tuning-sft)).

You can modify:

### 1) PPO Configuration
- learning rate
- KL coefficient
- clip range
- batch sizes
- number of PPO epochs
- value function coefficient

### 2) Model / LoRA Configuration
- base model
- LoRA rank (`lora_r`)
- LoRA alpha
- LoRA dropout
- target modules

### 3) Framework-Specific Reward Weights

---

# Reward Engineering

Custom reward functions can be added.

Take existing rewards as reference to include new rewards (see `rewards/extra_rewards.py`)

New rewards should be registered in `hyperparams.json`

---

## Usage
## Fine-Tuning

For general Python generation:

```bash
accelerate launch --config-file config/accelerate.yml main.py --mode fine_tune --param ppo_code_gen --framework_params code_gen
```

For robotics:

```bash
accelerate launch --config-file config/accelerate.yml main.py --mode fine_tune --param ppo --framework_params robo
```
## Evaluate 
on RoboEval

```bash
accelerate launch --config-file config/accelerate.yml main.py --mode evaluate_roboeval --param ppo --checkpoint checkpoint-XXX
```

Pass@K (EvalPlus / MBPP)

```bash
accelerate launch --config-file config/accelerate.yml main.py --mode evaluate_passk --param ppo_code_gen --checkpoint checkpoint-XXX
```

---

## Supervised Fine-Tuning (SFT)

A supervised fine-tuning baseline to compare against the PPO results. It follows the SFT recipe of the Robo-Instruct paper (LoRA, learning rate 3e-5, constant schedule with 3% warmup, AdamW, per-device batch 2 × gradient accumulation 4, 5 epochs, 2048-token sequences) using TRL's `SFTTrainer`. It lives in its own package, `sft/`, separate from `main.py`, `utils.py` and `wrappers/`, so it can't affect the RL pipeline.

### Training

Recipes are defined in `sft/sft_hyperparams.json`. The top-level `ScriptArguments` / `ModelConfig` / `SFTConfig` keys are the default recipe (robotics, Qwen2.5-Coder-1.5B-Instruct). Other domain/model-size combinations are named blocks, selected with `--variant`:

```bash
python -m sft.train_sft                             # robotics, 1.5B (default recipe)
python -m sft.train_sft --variant code_gen_1.5b      # Python (OpenCodeInstruct), 1.5B
python -m sft.train_sft --variant code_gen_7b        # Python (OpenCodeInstruct), 7B
```

- Run it as a module (`python -m sft.train_sft`) from the repo root. `python sft/train_sft.py` fails with `ModuleNotFoundError: No module named 'sft'`.
- No `accelerate launch` is needed. The Trainer uses every GPU visible to the process, so the effective batch size is 8 × the number of visible GPUs. Use `CUDA_VISIBLE_DEVICES` to control it (e.g. 6 GPUs give 48, which is 105 steps per epoch on the 5000-example OpenCodeInstruct split).
- `--max_steps N` overrides the epoch count (useful for a 1-step smoke test). `--adapter_path <.../policy>` resumes from an existing adapter.
- The OpenCodeInstruct variants use 5000 training and 500 test examples, filtered to `domain == "generic"` with non-empty unit tests. The robotics recipe uses an 80/20 split of `zichao22/robo-instruct`.
- A checkpoint is saved after every epoch, at `savings/<model>/<dataset>/sft/[<size>/]checkpoint-N/policy/`. The code-gen variants add a `1.5b` / `7b` folder so results for different model sizes don't overwrite each other.
- Loss is computed over the whole formatted text (system prompt, instruction and answer), not only the answer. As a result, trained models tend to restate the instruction before answering.

### Evaluation

SFT checkpoints are evaluated with `main.py`, the same way as PPO checkpoints, by passing the checkpoint's path under `sft/`:

```bash
python main.py --mode evaluate_roboeval --param ppo --framework_params robo --checkpoint sft/checkpoint-N
python main.py --mode evaluate_passk --param sft_code_gen_1.5b --checkpoint sft/1.5b/checkpoint-N
python main.py --mode evaluate_passk --param sft_code_gen_7b --checkpoint sft/7b/checkpoint-N
```

`--param` accepts any block name in `config/hyperparams.json`. It only tells `main.py` which base model and dataset to use, and that pair determines where the adapter is loaded from (`savings/<model>/<dataset>/<checkpoint>/policy`). The block must therefore match what the checkpoint was trained on. The robotics command above assumes `ppo` still points to Qwen2.5-Coder-1.5B-Instruct and `zichao22/robo-instruct`. If it doesn't, the adapter path is wrong and PEFT fails with a missing `adapter_config.json` error.

The published 7B Robo-Instruct SFT model (`zichao22/RI-FT-Qwen-Coder`) is a fully merged model, not an adapter, so it's evaluated without `--checkpoint`:

```bash
python main.py --mode evaluate_roboeval --param sft_robo_7b_reference --framework_params robo
```

Notes on evaluation:

- **Output locations:** RoboEval results go to `roboeval/<model>/<checkpoint>/` (or `.../original/` without `--checkpoint`). EvalPlus results go to `evalplus_passk/<checkpoint>/`: the scores are in `evalplus_run_log.json`, the extracted code in `evalplus_samples.jsonl`, and the raw model output before code extraction in `evalplus_raw_completions.jsonl`.
- **Re-running EvalPlus reuses old samples.** If `evalplus_samples.jsonl` already exists, generation is skipped and the existing samples are only re-scored. Delete the checkpoint's `evalplus_passk/...` folder before re-running after a config change.
- **Only fenced code is scored.** Code is taken from the first `` ```python `` block in the output. An answer without that fence becomes an empty completion.
- **Generation budget:** the number of generated tokens comes from `response_length` in the selected block. The `sft_code_gen_*` blocks use 1024, while the PPO blocks (`ppo`, `ppo_code_gen`) use 256, so their EvalPlus numbers come from different generation budgets. The prompt itself is truncated to 1024 tokens (`MAX_LENGTH` in `eval/evalplus_pass_k.py`).
- **Generation doesn't stop at the end of the answer.** It always runs to `response_length`, so a larger budget makes evaluation proportionally slower.
- **Memory:** EvalPlus generates in batches of 16 (in `eval/evalplus_pass_k.py`), which fits a 7B model at a 1024-token budget on one 80 GB GPU. Lower it if you hit out-of-memory errors. This only affects speed and memory, not results.
- **RoboEval timeout:** each simulated program gets 10 seconds (`rewards/robo_instruct/roboeval/benchmark/simple_tracer.py`). The original 1 second caused correct programs to time out and be counted as `PythonError`.

---

## Comparing Results

`analysis/compare_results.py` plots comparisons across any number of evaluated checkpoints (e.g. SFT vs. PPO, or different model sizes), in one of two modes depending on the benchmark:

**RoboEval** (`--result`): pass@1 comparison + outcome-breakdown chart (Success / CompletionError / RobotExecutionError / PythonError), reading from the `pass1/result.csv` and `error_breakdown/result.csv` files RoboEval evaluation produces:

```bash
python analysis/compare_results.py \
  --result "PPO=roboeval/Abgabe/Qwen-Abgabe/Qwen2.5-Coder-1.5B-Instruct/checkpoint-1200" \
  --result "SFT=roboeval/Qwen/Qwen2.5-Coder-1.5B-Instruct/sft/checkpoint-1250" \
  --out comparison.pdf
```

**EvalPlus / MBPP** (`--evalplus-result`): grouped MBPP/MBPP+ pass@1 bar chart, reading from the `evalplus_run_log.json` file `evaluate_passk` produces:

```bash
python analysis/compare_results.py \
  --evalplus-result "PPO=evalplus_passk/checkpoint-938" \
  --evalplus-result "SFT=evalplus_passk/sft/1.5b/checkpoint-525" \
  --out comparison.pdf
```

Pass exactly one of `--result` / `--evalplus-result` per run (they produce differently-shaped charts). Add more `LABEL=PATH` pairs to put additional methods or model sizes on the same chart.

