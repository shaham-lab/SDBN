# Generative Pipeline — SDBN-P

Adversarial PEFT training and robustness evaluation for decoder-only generative QA models (LLaMA, Qwen) using **pre-loaded adversarial paraphrases**.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Fine-tune a decoder-only LM with SDBN-P adversarial training |
| `eval.py` | Load trained models and evaluate on clean and noisy inputs |

---

## Supported Configurations

**Models** (`--model_name` / `--model`):

| Key | HuggingFace ID |
|-----|---------------|
| `llama` | `meta-llama/Llama-3.2-1B` |
| `llama-7b` | `meta-llama/Llama-2-7b-hf` |
| `qwen` | `Qwen/Qwen2.5-7B` |

**PEFT method** (`--peft_method`): `lora` (default), `none`

**Datasets** (`--dataset`): `squad`, `tweetqa`

**Perturbation**: `sdbn-p` — selects from pre-generated adversarial paraphrase variants (from CSV files) the one that maximises the model's loss.

---

## Adversarial Data Format

SDBN-P requires pre-generated CSV files with adversarial paraphrases for each training example. Each CSV must have the columns:

```
input, answer, input_adv1, input_adv2, input_adv3, input_adv4, input_adv5
```

Files must follow this naming convention:
```
trainset_seed{S}_sdbn_p.csv       # single CSV per run
trainset_seed{S}_sdbn_p_e{E}.csv  # epoch-specific CSV (--multi_csv mode)
```

Place them in:
```
{root}/sdbn-p_data/{dataset}/trainset_{train_size}/
```

Or pass `--resource_dir /path/to/csv/directory` to specify the location directly.

---

## Training (`train.py`)

```bash
python generative/train.py [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `squad` | Dataset: `squad` or `tweetqa` |
| `--model_name` | `llama` | Model key (see table above) |
| `--peft_method` | `lora` | PEFT method: `lora` or `none` |
| `--lora_rank` | `4` | LoRA rank |
| `--train_size`, `-tr` | `0` (full) | Number of training examples (0 = use all) |
| `--epochs`, `-e` | `5` | Number of adversarial training epochs |
| `--warmup_epochs`, `-we` | `1` | Clean warm-up epochs before adversarial phase |
| `--learning_rate`, `-lr` | `1e-4` | Learning rate |
| `--batch_train_size` | `16` | Training batch size |
| `--seed` | `0` | Random seed |
| `--device` | `0` | CUDA device index |
| `--resource_dir` | `None` | Directory containing adversarial CSV files |
| `--multi_csv` | `False` | Use epoch-specific CSV files |
| `--max_length` | `128` | Max input token length |
| `--max_target_length` | `32` | Max target token length |
| `--refs_per_ex` | `4` | Number of adversarial variants to sample per example |
| `--root` | `""` (cwd) | Root directory for data and model storage |
| `--not_overwrite` | `False` | Skip training if final model already exists |
| `--output_dir` | `None` | Override auto-resolved output directory |

### Output

Models are saved to:
```
{root}/results/SDBN-P/{dataset}/{model}/trainset_size_{train_size}/{peft_method}/rank_{lora_rank}/
    model_seed_{seed}.pt
```

### Example

```bash
cd /path/to/data/root
python /path/to/new_repo/generative/train.py \
  --dataset squad --model_name llama --lora_rank 4 \
  --train_size 200 --epochs 5 --warmup_epochs 1 \
  --resource_dir ./sdbn-p_data/squad/trainset_200 \
  --seed 0
```

Run multiple seeds by repeating with `--seed 1`, `--seed 2`, etc.

---

## Evaluation (`eval.py`)

Evaluates saved models on clean and noise-corrupted inputs. Reports Exact Match (EM) and F1.

```bash
python generative/eval.py [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `squad` | Dataset to evaluate on |
| `--model` | `llama` | Model key (see table above) |
| `--peft_method` | `lora` | PEFT method used during training |
| `--lora_rank` | `4` | LoRA rank used during training |
| `--train_size` | `1000` | Training set size used (used to locate saved models) |
| `--noise_type`, `-nt` | `replace_word` | Noise type to apply at eval: `delete_word`, `swap_word`, `replace_word`, `delete_char`, `swap_char`, `double_char`, `keyboard_char`, `homophone`, `sms`, `cyrillic`, `case` |
| `--noise_levels` | `1` | Number of noise intensity levels |
| `--seeds` | `5` | Number of seed models to evaluate |
| `--num_samples` | `200` | Number of test examples to evaluate |
| `--max_gen_len` | `32` | Max generation length |
| `--batch_size` | `8` | Eval batch size |
| `--root` | `""` (cwd) | Root directory where models are stored |
| `--device` | `0` | CUDA device index |

### Example

```bash
cd /path/to/data/root
python /path/to/new_repo/generative/eval.py \
  --dataset squad --model llama --lora_rank 4 \
  --train_size 200 --seeds 5 \
  --noise_type delete_word --noise_levels 1
```

---

## Notes

- LLaMA models require a HuggingFace token. Set `HUGGINGFACE_HUB_TOKEN` in your environment before running.
- The 7B models (`llama-7b`, `qwen`) are loaded in `bfloat16` automatically.
- `utils_gen.py` in `utils/` provides the `load_lora_gen_model` function used by `eval.py` for loading saved LoRA checkpoints.
