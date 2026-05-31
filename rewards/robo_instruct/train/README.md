# 🛠️ Environment Setup
<!-- TODO -->
We use [Unsloth](https://github.com/unslothai) to accelerate training. However, Unsloth introduces dependency conflicts with the packages required for `robo-instruct` evaluation.
To avoid these issues, we recommend using a separate Conda environment for training:

```bash
conda create --name robo-instruct-train \
  python=3.10 \
  pytorch \
  pytorch-cuda=12.1 \
  cudatoolkit \
  xformers \
  -c pytorch -c nvidia -c xformers -y
```

Activate the environment:

```bash
conda activate robo-instruct-train
```

CD into the `train/` directory and install the training dependencies:

```bash
cd train/
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r train_requirements.txt 
```

# 🚀 Training
We use Hydra for configuration management. All training configs are stored in the config/ directory.

To start training:

```bash
CUDA_VISIBLE_DEVICES=0 python train_peft.py +exps=llama_exp_ri
```

After training, merge the PEFT model with the base model:

```bash
python merge_peft.py \
  --model_path <path_to_trained_peft_model> \
  --save_path <path_to_save_merged_model>
```
Replace <path_to_trained_peft_model> and <path_to_save_merged_model> with your actual paths.