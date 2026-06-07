# Classification Pipeline — SDBN / SDBN-H

Adversarial PEFT training and robustness evaluation for text classification tasks.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Fine-tune a classifier with SDBN or SDBN-H adversarial training |
| `eval.py` | Load trained models and evaluate robustness under various noise types |

---

## Supported Configurations

**Models** (`--model`):

| Key | HuggingFace ID |
|-----|---------------|
| `bert-b` | `bert-base-uncased` |
| `deberta` | `microsoft/deberta-v3-base` |
| `deberta-l` | `microsoft/deberta-v3-large` |
| `mbert` | `bert-base-multilingual-uncased` |
| `mdeberta` | `microsoft/mdeberta-v3-base` |

**PEFT methods** (`--alg`): `lora`, `qlora`, `bitfit`, `adapter`, `full_ft`

**Datasets** (`--dataset`): `banking77`, `trec`, `imdb`, `nli`, `news`, `bless`, `ArSarcasm-v2`

**Perturbations** (`--pertubation`):
- `sdbn` — embedding-space FGSM adversarial perturbation
- `sdbn-h` — hybrid: SDBN + character-level noise injection

---

## Training (`train.py`)

Run from the data root directory (where `banking77/` etc. reside).

```bash
python classification/train.py [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `banking77` | Dataset to use |
| `--model` | `bert-b` | Model key (see table above) |
| `--alg` | `lora` | PEFT method |
| `--rank` | `12` | LoRA rank (use 12 for BERT, 4 for DeBERTa) |
| `--pertubation`, `-pt` | `sdbn` | Perturbation type: `sdbn` or `sdbn-h` |
| `--epochs` | `10` | Number of adversarial training epochs |
| `--warm_up` | `3` | Clean warm-up epochs before adversarial phase |
| `--train_set_size` | `1.0` | Fraction of training data to use (e.g. `0.1` = 10%) |
| `--lr` | `1e-4` | Learning rate |
| `--batch_size` | `32` | Training batch size |
| `--epsilon`, `-ep` | `1e-4` | Adversarial perturbation magnitude |
| `--init_seed` | `0` | Random seed |
| `--device` | `0` | CUDA device index |
| `--output`, `-o` | `""` | Optional suffix for the output directory name |

### Output

Models are saved to:
```
./results/SDBN/Ep_{epochs}/{dataset}/percent_{train_set_size}/{model}/{alg}_{pertubation}/rank_{rank}/
    weights_baseline_adv_seed{seed}_epoch_final.pt
```

### Example

```bash
cd /path/to/data/root
python /path/to/new_repo/classification/train.py \
  --dataset banking77 --model deberta --alg lora --rank 4 \
  --epochs 10 --warm_up 3 --train_set_size 0.1 --pertubation sdbn
```

---

## Evaluation (`eval.py`)

Loads saved models and evaluates accuracy on clean and noise-corrupted test sets.

```bash
python classification/eval.py [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `banking77` | Dataset to evaluate on |
| `--model` | `deberta` | Model key |
| `--baseline` | *(all)* | Restrict to a single PEFT method (e.g. `lora`) |
| `--pertubation`, `-pt` | *(all)* | Restrict to one perturbation type |
| `--rank` | `4` | LoRA rank used during training |
| `--epochs` | `10` | Number of epochs used during training (used to locate saved models) |
| `--percent` | `0.1` | Training set fraction used during training |
| `--noise_type`, `-nt` | `None` | Noise to apply at eval: `delete_char`, `swap_char`, `double_char`, `keyboard_char`, `delete_word`, `swap_word`, `replace_word`, `homophone`, `sms`, `cyrillic`, `case`, `pronoun_swap`, `add_space`, `remove_space` |
| `--noise_levels` | `3` | Number of noise intensity levels to test |
| `--seeds` | `5` | Number of seed models expected per experiment |
| `--device` | `0` | CUDA device index |

### Example

```bash
cd /path/to/data/root
python /path/to/new_repo/classification/eval.py \
  --dataset banking77 --model deberta --baseline lora --rank 4 \
  --epochs 10 --percent 0.1 --pertubation sdbn \
  --noise_type delete_char --seeds 5
```

---

## Notes

- **Banking77** requires `banking77/train.csv` and `banking77/test.csv` in the working directory.
- **ArSarcasm-v2** uses multilingual BERT (`mbert`) and requires `--task` to specify the dialect (e.g. `sentiment_egypt`).
- Run multiple seeds by repeating training with `--init_seed 0`, `--init_seed 1`, etc., then run eval with `--seeds N`.
