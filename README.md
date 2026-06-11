
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
Install syncode seperately 

```bash
pip install --no-dep syncode==0.4.16
```

If using DeepSpeed, ensure compatibility with your CUDA and PyTorch versions.

---

# Configuration

All hyperparameters are defined in:

```
hyperparams.json
```

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

