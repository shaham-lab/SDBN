# Small Data, Big Noise: Adversarial Training for Robust Parameter-Efficient Fine-Tuning

This repository contains the code for the paper **"Small Data, Big Noise: Adversarial Training for Robust Parameter-Efficient Fine-Tuning"**.

We propose **SDBN** — an adversarial training framework that makes PEFT methods (LoRA, QLoRA, BitFit, Adapter, full fine-tuning) robust to input noise when trained on small datasets. Two pipelines are provided:

| Pipeline | Method | Models | Datasets |
|---|---|---|---|
| **Classification** | SDBN / SDBN-H | BERT, DeBERTa, mBERT | Banking77, TREC, IMDB, NLI, News, BLESS, ArSarcasm-v2 |
| **Generative** | SDBN-P | LLaMA-3.2-1B, LLaMA-2-7B, Qwen2.5-7B | SQuAD, TweetQA |

---

## Repository Structure

```
new_repo/
├── classification/
│   ├── train.py          # Adversarial PEFT training (SDBN / SDBN-H)
│   └── eval.py           # Robustness evaluation under noise
├── generative/
│   ├── train.py          # Adversarial PEFT training with pre-loaded variants (SDBN-P)
│   └── eval.py           # Robustness evaluation under noise
├── utils/
│   ├── utils.py          # Shared utilities for classification
│   └── utils_gen.py      # Shared utilities for generative
├── requirements.txt
└── run.sh                # Helper to invoke the conda environment's Python
```

---

## Setup

### 1. Create environment

```bash
conda create -n sdbn python=3.9
conda activate sdbn
pip install -r requirements.txt
```

> **Note:** `adapters` is a development install. Install it from source if the pip version is unavailable:
> ```bash
> pip install git+https://github.com/adapter-hub/adapters.git
> ```

### 2. Using `run.sh`

`run.sh` is a convenience wrapper that invokes the correct Python environment without requiring `conda activate`:

```bash
chmod +x run.sh
./run.sh classification/train.py [args...]
./run.sh generative/train.py [args...]
```

---

## Quick Start

Scripts must be run from the **data root directory** (where dataset folders such as `banking77/` reside).

### Classification

```bash
# Train (SDBN, LoRA, DeBERTa, Banking77)
./run.sh classification/train.py \
  --dataset banking77 --model deberta --alg lora --rank 4 \
  --epochs 10 --warm_up 3 --train_set_size 0.1 --pertubation sdbn

# Evaluate
./run.sh classification/eval.py \
  --dataset banking77 --model deberta --baseline lora --rank 4 \
  --epochs 10 --percent 0.1 --pertubation sdbn --noise_type delete_char
```

### Generative

Pre-generated adversarial CSV files are required (see [generative/README.md](generative/README.md)).

```bash
# Train (SDBN-P, LoRA, LLaMA-3.2-1B, SQuAD)
./run.sh generative/train.py \
  --dataset squad --model_name llama --lora_rank 4 \
  --train_size 200 --epochs 5 --warmup_epochs 1 \
  --resource_dir /path/to/sdbn-p_data/squad/trainset_200

# Evaluate
./run.sh generative/eval.py \
  --dataset squad --model llama --lora_rank 4 \
  --train_size 200 --noise_type delete_word
```

---

## Perturbation Methods

| Name | Description | Pipeline |
|------|-------------|----------|
| `sdbn` | Embedding-space L∞ FGSM adversarial perturbation | Classification |
| `sdbn-h` | Hybrid: SDBN + character-level noise | Classification |
| `sdbn-p` | Pre-loaded adversarial paraphrases (5 variants per example) | Generative |

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{sdbn2024,
  title={Small Data, Big Noise: Adversarial Training for Robust Parameter-Efficient Fine-Tuning},
  author={...},
  year={2024}
}
```
