import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
    TrainingArguments
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np
from tqdm import tqdm
import json
from collections import Counter
import string
import re
from typing import Dict, List, Tuple
import argparse
import warnings
import utils
import utils_gen
import os
import sacrebleu

models_dict = {
    'bert' : 'bert-base-uncased',
    'llama' : 'meta-llama/Llama-2-7b-hf',
    'mbert' : 'bert-base-multilingual-uncased',
    'deberta' : 'microsoft/deberta-v3-base',
    'deberta-l' : "microsoft/deberta-v3-large",
    'mdeberta' : 'microsoft/mdeberta-v3-base',
    # Seq2seq models
    't5':           'google/flan-t5-base',
    'flan-t5-base': 'google/flan-t5-base'
}

warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser(description='Fine-tune DeBERTa-v3 on SQuAD with LoRA and optional adversarial training')
parser.add_argument('--perturbation', '-pt', type=str, choices=[None, 'sdbn', 'neftune'], default=None,
                    help='Type of adversarial perturbation to use')
parser.add_argument('--warmup_epochs', '-we', type=int, default=1,
                    help='Number of warmup epochs with normal training')
parser.add_argument('--epochs', '-e', type=int, default=5,
                    help='Number of epochs after warmup (normal or adversarial based on perturbation)')
parser.add_argument('--epsilon', '-eps', type=float, default=5e-4,
                    help='Perturbation size for SDBN |  1e-2 for SQUAD')
parser.add_argument('--neftune_alpha', '-na', type=float, default=5,
                    help='Scaling factor for NEFTune noise')
parser.add_argument('--model_name', type=str, default='deberta',
                    help='DeBERTa-v3 model to fine-tune')
parser.add_argument('--lora_rank', type=int, default=4,
                    help='LoRA rank (default: 4)')

parser.add_argument("--output_dir", type=str, default=None, help="Directory to save model and checkpoints")
parser.add_argument("--batch_train_size", type=int, default=16)
parser.add_argument("--batch_val_size", type=int, default=32)
parser.add_argument("--learning_rate", '-lr', type=float, default=1e-4, help="Learning rate for training 1e-4 for SQUAD, 1e-3 for Tatoeba")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
parser.add_argument("--device", type=int, default=0, help="GPU device number (e.g., 0, 1)")
parser.add_argument("--dataset", type=str, default="squad", help="Dataset to use (e.g., 'squad')")
parser.add_argument("--train_size", '-tr', type=int, default=200, help="Limit training set size (e.g., 1000)")
parser.add_argument("--val_size", '-va', type=int, default=1000, help="Limit validation set size (e.g., 1000)")
parser.add_argument("--public_server", "-ps", default=False, action="store_true", help="Save the model on my directory")
parser.add_argument("--peft_method", type=str, default="lora", choices=["lora", "none"], help="PEFT method to use")
args = parser.parse_args()

utils.set_random_seed(args.seed)  # Set random seed for reproducibility

if args.public_server:
    root = os.environ['WA']
else:
    root = os.environ['STORE']

model_full_name = models_dict[args.model_name]
output_dir = os.path.join(root, 'results', 'ALoRA', args.dataset, args.model_name, f"trainset_size_{args.train_size}", args.peft_method, args.perturbation if args.perturbation else "vanilla")

# Calculate total epochs
total_epochs = args.warmup_epochs + args.epochs

# Set device
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Training configuration:")
print(f"  Model: {model_full_name}")
print(f"  LoRA rank: {args.lora_rank}")
print(f"  Warmup epochs (normal training): {args.warmup_epochs}")
print(f"  Main epochs: {args.epochs}")
print(f"  Total epochs: {total_epochs}")
print(f"  Perturbation: {args.perturbation}")
if args.perturbation:
    print(f"  Adversarial training will be used for the last {args.epochs} epochs")
    if args.perturbation == 'sdbn':
        print(f"  SDBN epsilon: {args.epsilon}")
    elif args.perturbation == 'neftune':
        print(f"  NEFTune alpha: {args.neftune_alpha}")
else:
    print(f"  All {total_epochs} epochs will use normal training")

# -------------------------------------------------------------
# Optional path: Machine Translation evaluation (Tatoeba es→pt)
# Model: FLAN-T5-BASE, Metric: sacreBLEU
# -------------------------------------------------------------
if args.dataset.replace("_", "-").lower() == "tatoeba-es-pt":
    # ---------- helpers ----------
    def render_prompt_es2pt(s: str) -> str:
        return f"translate Spanish to Portuguese: {s}"

    @torch.no_grad()
    def translate_batch(texts, tok, model, device, max_new_tokens: int = 32):
        x = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        y = model.generate(
            **x,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        return tok.batch_decode(y, skip_special_tokens=True)

    # ---------- adversarial helpers for seq2seq (encoder-side) ----------
    def perform_adversarial_attack_seq2seq(model, input_ids, attention_mask, labels,
                                           perturbation_type='sdbn', epsilon=5e-4, neftune_alpha=5):
        # Build input embeddings with grad
        emb_w = model.get_input_embeddings().weight
        embeds = torch.nn.functional.embedding(input_ids, emb_w)
        embeds = embeds.clone().detach().requires_grad_(perturbation_type == 'sdbn')

        if perturbation_type == 'sdbn':
            # Compute grad wrt embeddings only
            outputs = model(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            grad = torch.autograd.grad(loss, embeds, retain_graph=False, create_graph=False)[0]
            delta = torch.sign(grad) * float(epsilon)
        elif perturbation_type == 'neftune':
            # Uniform noise scaled per NEFTune
            input_mask = attention_mask.unsqueeze(-1).to(embeds.dtype)
            noise = torch.zeros_like(embeds).uniform_(-1, 1) * input_mask
            lengths = attention_mask.sum(1)
            dims = lengths * embeds.size(-1)
            mag = float(neftune_alpha) / torch.sqrt(dims.float())
            delta = (noise * mag.view(-1, 1, 1)).detach()
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

        adv_embeds = embeds.detach() + delta
        return adv_embeds

    # ---------- (1) load dataset ----------
    ds = load_dataset("tatoeba", "es-pt", trust_remote_code=True)

    # unwrap the `translation` dict → (src, tgt)
    def to_src_tgt(ex):
        t = ex["translation"]
        return {"src": t["es"], "tgt": t["pt"]}

    # Always build disjoint pools from the original train split:
    # - First half: train potential pool
    # - Second half: eval potential pool
    full_raw = ds["train"].map(to_src_tgt, remove_columns=ds["train"].column_names)
    N = len(full_raw)
    mid = N // 2
    pool_train_idx = list(range(0, mid))
    pool_eval_idx = list(range(mid, N))
    rng = np.random.RandomState(args.seed)
    # desired sizes (clamped to pool sizes)
    desired_train = args.train_size if args.train_size else len(pool_train_idx)
    desired_val = args.val_size if args.val_size else len(pool_eval_idx)
    k_train = min(desired_train, len(pool_train_idx))
    k_val = min(desired_val, len(pool_eval_idx))
    sel_train = list(np.array(pool_train_idx)[rng.permutation(len(pool_train_idx))[:k_train]])
    sel_eval = list(np.array(pool_eval_idx)[rng.permutation(len(pool_eval_idx))[:k_val]])
    # Keep sorted for stable ordering (optional)
    sel_train.sort()
    sel_eval.sort()
    train_raw = full_raw.select(sel_train)
    val_raw = full_raw.select(sel_eval)
    created_split_note = " (half-split pools; sampled per seed)"

    # Verification and debug prints
    print(f"Tatoeba pools: train_pool=[0,{mid-1}] (n={len(pool_train_idx)}), eval_pool=[{mid},{N-1}] (n={len(pool_eval_idx)})")
    if len(sel_train) > 0:
        print(f"Selected train idx range: min={min(sel_train)}, max={max(sel_train)} (k={len(sel_train)})")
        assert max(sel_train) < mid, "Train selection should come only from the first half"
    if len(sel_eval) > 0:
        print(f"Selected eval idx range: min={min(sel_eval)}, max={max(sel_eval)} (k={len(sel_eval)})")
        assert min(sel_eval) >= mid, "Eval selection should come only from the second half"
    assert set(sel_train).isdisjoint(sel_eval), "Train/Eval selections must be disjoint"

    print("Dataset: Tatoeba (es→pt)")
    print(ds)
    print(f"\nTrain size: {len(train_raw)}, Val size: {len(val_raw)}{created_split_note}")

    # ---------- (2) build datasets ----------
    # Resolve model from shared --model_name arg (falls back to raw if not in dict)
    model_name = models_dict.get(args.model_name, args.model_name)
    tok = AutoTokenizer.from_pretrained(model_name)

    class MTSeq2SeqDataset(Dataset):
        def __init__(self, hf_ds, tokenizer, max_source_len=128, max_target_len=128):
            self.hf_ds = hf_ds
            self.tok = tokenizer
            self.max_source_len = max_source_len
            self.max_target_len = max_target_len

        def __len__(self):
            return len(self.hf_ds)

        def __getitem__(self, idx):
            ex = self.hf_ds[idx]
            src = ex["src"]
            tgt = ex["tgt"]
            prompt = render_prompt_es2pt(src)
            x = self.tok(
                prompt,
                truncation=True,
                max_length=self.max_source_len,
                padding="max_length",
            )
            y = self.tok(
                text_target=tgt,
                truncation=True,
                max_length=self.max_target_len,
                padding="max_length",
            )
            labels = y["input_ids"]
            labels = [(-100 if t == self.tok.pad_token_id else t) for t in labels]
            return {
                "input_ids": torch.tensor(x["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(x["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    max_src_len = 128
    max_tgt_len = 128
    train_ds = MTSeq2SeqDataset(train_raw, tok, max_src_len, max_tgt_len)
    val_ds = MTSeq2SeqDataset(val_raw, tok, max_src_len, max_tgt_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_train_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_val_size, shuffle=False)

    # ---------- (3) model (optional LoRA) ----------
    model_mt = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if args.peft_method == "lora":
        # Typical target modules for T5
        lcfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["q", "k", "v", "o", "wi", "wo"],
            lora_dropout=0.05,
            bias="none",
        )
        model_mt = get_peft_model(model_mt, lcfg)
        model_mt.print_trainable_parameters()

    device_mt = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model_mt = model_mt.to(device_mt)

    # ---------- (4) optimizer/scheduler ----------
    optimizer = torch.optim.AdamW(model_mt.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * (args.warmup_epochs + args.epochs)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ---------- (5) training loop with sacreBLEU eval ----------
    def evaluate_sacrebleu(model, tok, val_hf_ds, device, K=None):
        model.eval()
        K = min(K or len(val_hf_ds), len(val_hf_ds))
        refs = [[val_hf_ds[i]["tgt"].strip() for i in range(K)]]
        hyps: List[str] = []
        # Generation batch size follows --batch_val_size
        gen_bs = max(1, int(getattr(args, "batch_val_size", 16)))
        total_steps = (K + gen_bs - 1) // gen_bs
        pbar = tqdm(range(0, K, gen_bs), total=total_steps, desc="MT Eval", leave=False)
        for start in pbar:
            end = min(start + gen_bs, K)
            prompts = [render_prompt_es2pt(val_hf_ds[i]["src"]) for i in range(start, end)]
            batch_out = translate_batch(prompts, tok, model, device)
            hyps.extend([h.strip() for h in batch_out])
            pbar.set_postfix({"done": f"{end}/{K}"})
            torch.cuda.empty_cache()
        return sacrebleu.corpus_bleu(hyps, refs).score

    best_bleu = -1.0
    history = {"epoch": [], "train_loss": [], "val_sacrebleu": []}

    num_epochs = args.warmup_epochs + args.epochs
    print(f"\nStarting MT training for {num_epochs} epochs on Tatoeba es→pt...")
    for epoch in range(num_epochs):
        model_mt.train()
        total_loss = 0.0
        # Determine training mode
        use_adversarial = (epoch >= args.warmup_epochs) and (args.perturbation is not None)
        mode_str = f"adv({args.perturbation})" if use_adversarial else "normal"
        pbar = tqdm(train_loader, desc=f"MT Training (epoch {epoch+1}/{num_epochs}) [{mode_str}]")
        for batch in pbar:
            batch = {k: v.to(device_mt) for k, v in batch.items()}
            if use_adversarial:
                adv_embeds = perform_adversarial_attack_seq2seq(
                    model_mt,
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['labels'],
                    perturbation_type=args.perturbation,
                    epsilon=args.epsilon,
                    neftune_alpha=args.neftune_alpha
                )
                out = model_mt(
                    inputs_embeds=adv_embeds,
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
            else:
                out = model_mt(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_mt.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "mode": mode_str})

        avg_loss = total_loss / max(1, len(train_loader))
        bleu = evaluate_sacrebleu(model_mt, tok, val_raw, device_mt, K=min(args.val_size, len(val_raw)))
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, val_sacreBLEU={bleu:.2f}")

        history["epoch"].append(epoch+1)
        history["train_loss"].append(avg_loss)
        history["val_sacrebleu"].append(bleu)

        # Save best
        if bleu > best_bleu:
            best_bleu = bleu
            # Build output_dir consistent with repo layout (use shared convention)
            if args.public_server:
                root_dir = os.environ['WA']
            else:
                root_dir = os.environ['STORE']
            out_dir = os.path.join(
                root_dir,
                'results', 'ALoRA', args.dataset, args.model_name, f"trainset_size_{args.train_size}",
                args.peft_method, args.perturbation if args.perturbation else 'vanilla'
            )
            os.makedirs(out_dir, exist_ok=True)
            ckpt_dir = os.path.join(out_dir, 'best')
            os.makedirs(ckpt_dir, exist_ok=True)
            # Save HF checkpoint
            model_mt.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            # Save metrics
            with open(os.path.join(out_dir, 'mt_history.json'), 'w') as f:
                json.dump(history, f, indent=2)
            print(f"[✓] Saved best checkpoint to {ckpt_dir} (sacreBLEU={best_bleu:.2f})")

    print(f"\nTraining complete. Best sacreBLEU: {best_bleu:.2f}")

    # ---------- (6) Save FINAL model (last epoch) and verify ----------
    # Build output_dir consistent with repo layout (use shared convention)
    if args.public_server:
        root_dir = os.environ['WA']
    else:
        root_dir = os.environ['STORE']
    out_dir = os.path.join(
        root_dir,
        'results', 'ALoRA', args.dataset, args.model_name, f"trainset_size_{args.train_size}",
        args.peft_method, args.perturbation if args.perturbation else 'vanilla'
    )
    os.makedirs(out_dir, exist_ok=True)
    final_dir = os.path.join(out_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)

    # Evaluation uses the second-half pool selection (val_raw) to guarantee disjointness
    eval_raw = val_raw

    # Evaluate LAST model on eval split to record reference BLEU
    final_K = min(args.val_size, len(eval_raw))
    final_bleu_ref = evaluate_sacrebleu(model_mt, tok, eval_raw, device_mt, K=final_K)
    print(f"\nFINAL epoch sacreBLEU on eval split ({final_K}): {final_bleu_ref:.4f}")

    # Save final model checkpoint
    model_mt.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    with open(os.path.join(final_dir, 'final_bleu.txt'), 'w') as f:
        f.write(f"{final_bleu_ref:.8f}\n")

    # Reload the saved final model and re-evaluate
    try:
        base_reload = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        base_reload = base_reload.to(device_mt)
        base_reload.eval()
        if args.peft_method == 'lora':
            reloaded = PeftModel.from_pretrained(base_reload, final_dir)
        else:
            # No LoRA was used; load directly from final_dir
            reloaded = AutoModelForSeq2SeqLM.from_pretrained(final_dir).to(device_mt).eval()
        bleu_reloaded = evaluate_sacrebleu(reloaded, tok, eval_raw, device_mt, K=final_K)
        print(f"Reloaded FINAL sacreBLEU on eval split: {bleu_reloaded:.4f}")
        # Compare with small tolerance
        tol = 1e-6
        if abs(bleu_reloaded - final_bleu_ref) > tol:
            print(f"[ERROR] Reloaded BLEU ({bleu_reloaded:.6f}) differs from last-epoch BLEU ({final_bleu_ref:.6f}) > tol={tol}")
            import sys
            sys.exit(1)
        else:
            print("[✓] Verification passed: reloaded model matches last-epoch performance")
    except Exception as e:
        print(f"[ERROR] Verification failed with exception: {e}")
        import sys
        sys.exit(1)

    # ---------- (7) Save per-seed artifacts in method dir ----------
    # Save the final result summary (BLEU) as quick .txt under method dir
    seed_bleu_txt = os.path.join(out_dir, f"mt_final_bleu_seed_{args.seed}.txt")
    with open(seed_bleu_txt, 'w') as f:
        f.write(f"{final_bleu_ref:.8f}\n")
    # Save full per-epoch history similar to classification artifacts
    # Includes: epoch indices, train_loss per epoch, val sacreBLEU per epoch, and evaluation sizes (K)
    epoch_eval_K = min(args.val_size, len(val_raw))
    seed_metrics_pt = os.path.join(out_dir, f"mt_final_bleu_seed_{args.seed}.pt")
    try:
        torch.save({
            "epoch": list(history.get("epoch", [])),
            "train_loss": list(history.get("train_loss", [])),
            "val_sacrebleu": list(history.get("val_sacrebleu", [])),
            "K": int(epoch_eval_K),
            "final_eval_K": int(final_K),
            "final_sacrebleu": float(final_bleu_ref)
        }, seed_metrics_pt)
    except Exception as _:
        pass

    # Save model weights with seed number in filename under method dir
    seed_model_path = os.path.join(out_dir, f"model_{args.seed}.pt")
    try:
        if args.peft_method == 'lora':
            # Save LoRA adapter weights via existing utils_gen helper for consistency across repo
            utils_gen.save_lora_gen_model(model_mt, seed_model_path)
        else:
            # Save full model state_dict for the non-LoRA case
            torch.save(model_mt.state_dict(), seed_model_path)
    except Exception as e:
        print(f"[WARN] Failed to save seed-suffixed model artifact: {e}")
    else:
        print(f"[✓] Saved per-seed artifacts: {os.path.basename(seed_bleu_txt)}, {os.path.basename(seed_model_path)}")

    # Exit after MT training/verification to keep QA code from running
    raise SystemExit(0)

class SQuADDataset(Dataset):
    """Custom Dataset for SQuAD"""
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        item = {}
        for key, val in self.encodings.items():
            if key == 'example_id':
                # Keep example_id as string
                item[key] = val[idx]
            else:
                # Convert numerical data to tensors
                item[key] = torch.tensor(val[idx])
        return item

def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    """Tokenize and prepare features for training"""
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]
    
    # Tokenize
    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # Process each feature
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)
        
        sample_index = sample_mapping[i]
        answer = answers[sample_index]
        
        if len(answer["answer_start"]) == 0:
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
        else:
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            
            if not (offsets[token_start_index][0] <= start_char and 
                    offsets[token_end_index][1] >= end_char):
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized["start_positions"].append(token_start_index - 1)
                
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized["end_positions"].append(token_end_index + 1)
    
    return tokenized

def prepare_validation_features(examples, tokenizer, max_length=384, doc_stride=128):
    """Prepare validation features"""
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    
    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []
    
    for i in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(i)
        context_index = 1
        
        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index])
        
        tokenized["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]
    
    return tokenized

def postprocess_qa_predictions(start_logits, end_logits, features, examples, n_best=20, max_answer_length=30):
    """Convert model outputs to answers"""
    all_start_logits = start_logits
    all_end_logits = end_logits
    
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = {}
    for i, feature in enumerate(features):
        features_per_example.setdefault(example_id_to_index[feature["example_id"]], []).append(i)
    
    predictions = {}
    
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []
        
        context = example["context"]
        for feature_index in feature_indices:
            start_logit = all_start_logits[feature_index]
            end_logit = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logit[cls_index] + end_logit[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score
            
            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logit[start_index] + end_logit[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        
        predictions[example["id"]] = best_answer["text"]
    
    return predictions

def get_embedding_weights(model):
    """Get embedding weights from DeBERTa-v3 model"""
    if hasattr(model, 'base_model'):
        # For PEFT models
        base_model = model.base_model.model
    else:
        base_model = model
    
    # DeBERTa-v3 specific
    if hasattr(base_model, 'deberta'):
        return base_model.deberta.embeddings.word_embeddings.weight
    else:
        raise ValueError("Model type not supported for embedding extraction")

def sdbn_perturbation(gradient, epsilon=5e-4):
    """SDBN perturbation - sign of gradient"""
    delta = torch.sign(gradient) * epsilon
    return delta

def neftune_perturbation(embeds, attention_mask, neftune_alpha=5):
    """NEFTune perturbation - uniform noise"""
    input_mask = attention_mask.unsqueeze(-1).to(embeds.dtype)  # B x L x 1
    input_lengths = torch.sum(attention_mask, 1)  # B
    
    noise = torch.zeros_like(embeds).uniform_(-1, 1)
    delta = noise * input_mask
    
    dims = input_lengths * embeds.size(-1)
    mag = neftune_alpha / torch.sqrt(dims.float())
    delta = (delta * mag.view(-1, 1, 1)).detach()
    
    return delta

def perform_adversarial_attack(model, input_ids, attention_mask, start_positions, end_positions, 
                             perturbation_type='sdbn', epsilon=5e-4, neftune_alpha=5):
    """
    Performs adversarial attack on DeBERTa-v3 for question answering
    """
    was_training = model.training
    
    model.eval()
    
    # Get embedding weights and create embeddings with gradients
    embedding_weights = get_embedding_weights(model)
    embeddings = torch.nn.functional.embedding(input_ids, embedding_weights)
    embeddings = embeddings.clone().detach().requires_grad_(True)
    
    # Apply perturbations based on type
    
    if perturbation_type == 'sdbn':
        # Forward pass with original embeddings
        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask,
                    start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        
        # Backward pass to get gradients
        loss.backward()
        
        gradient = embeddings.grad.clone()
    
    
    
        delta = sdbn_perturbation(gradient, epsilon=epsilon)
    elif perturbation_type == 'neftune':
        delta = neftune_perturbation(embeddings, attention_mask, neftune_alpha=neftune_alpha)
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")
    
    # Create adversarial embeddings
    adversarial_embeddings = embeddings.detach() + delta
    
    diff = adversarial_embeddings - embeddings.detach()
    #print(f"Adversarial perturbation stats: max={diff.max().item():.6f}, min={diff.min().item():.6f}")
    #print("diff:", diff)
    #exit(1)
    
    if was_training:
        model.train()
        
    return adversarial_embeddings

def train_one_epoch(model, train_dataloader, optimizer, scheduler, device, epoch, 
                   use_adversarial=False, perturbation_type='sdbn', 
                   epsilon=5e-4, neftune_alpha=5, max_grad_norm=1.0):
    """
    Train for one epoch with optional adversarial training
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Training (epoch {epoch})")
    
    training_mode = "adversarial" if use_adversarial else "normal"
    if use_adversarial:
        training_mode += f" ({perturbation_type})"
    
    for step, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if use_adversarial:
            # Adversarial training
            adversarial_embeds = perform_adversarial_attack(
                model, 
                batch['input_ids'], 
                batch['attention_mask'],
                batch['start_positions'],
                batch['end_positions'],
                perturbation_type=perturbation_type,
                epsilon=epsilon,
                neftune_alpha=neftune_alpha
            )
            # for making sure gradients are zero before backward pass
            optimizer.zero_grad()
            # Forward pass with adversarial embeddings
            outputs = model(
                inputs_embeds=adversarial_embeds,
                attention_mask=batch['attention_mask'],
                start_positions=batch['start_positions'],
                end_positions=batch['end_positions']
            )
        else:
            # Normal training
            outputs = model(**batch)
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix({'loss': loss.item(), 'mode': training_mode})
        
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

def compute_metrics(predictions, references):
    """Compute EM and F1 scores"""
    def normalize_answer(s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def compute_exact(a_gold, a_pred):
        return int(normalize_answer(a_gold) == normalize_answer(a_pred))
    
    def compute_f1(a_gold, a_pred):
        gold_toks = normalize_answer(a_gold).split()
        pred_toks = normalize_answer(a_pred).split()
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    exact_scores = []
    f1_scores = []
    
    for qid, gold_answer in references.items():
        if qid not in predictions:
            exact_scores.append(0)
            f1_scores.append(0)
            continue
        
        prediction = predictions[qid]
        exact_scores.append(compute_exact(gold_answer, prediction))
        f1_scores.append(compute_f1(gold_answer, prediction))
    
    return {
        'exact_match': 100.0 * sum(exact_scores) / len(exact_scores),
        'f1': 100.0 * sum(f1_scores) / len(f1_scores)
    }

def evaluate_model(model, eval_dataloader, eval_dataset_for_model, eval_examples):
    """Evaluate model on dataset"""
    model.eval()
    all_start_logits = []
    all_end_logits = []
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'example_id'}
            outputs = model(**inputs)
            
            all_start_logits.extend(outputs.start_logits.cpu().numpy())
            all_end_logits.extend(outputs.end_logits.cpu().numpy())
            torch.cuda.empty_cache()
    
    all_start_logits = np.array(all_start_logits)
    all_end_logits = np.array(all_end_logits)
    
    predictions = postprocess_qa_predictions(
        all_start_logits, 
        all_end_logits, 
        eval_dataset_for_model, 
        eval_examples
    )
    
    references = {}
    for example in eval_examples:
        if len(example['answers']['text']) > 0:
            references[example['id']] = example['answers']['text'][0]
        else:
            references[example['id']] = ""
    
    metrics = compute_metrics(predictions, references)
    return metrics

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_full_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_full_name)

# Configure LoRA for DeBERTa-v3
lora_config = LoraConfig(
    task_type=TaskType.QUESTION_ANS,
    r=args.lora_rank,  # Rank from args
    lora_alpha=args.lora_rank * 2,  # Common practice: alpha = 2 * rank
    target_modules=["query_proj", "key_proj", "value_proj", "dense"],  # DeBERTa-v3 attention layers
    lora_dropout=0.1,  # Dropout for regularization
    bias="none",
    modules_to_save=["qa_outputs"],  # Save the QA head
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = model.to(device)

# Load SQuAD dataset
dataset = load_dataset("squad")
if args.train_size:
    train_examples = dataset["train"].select(range(min(args.train_size, len(dataset["train"]))))
else:
    train_examples = dataset["train"]
print(f"Loaded {len(train_examples)} training examples from SQuAD")

if args.val_size:
    validation_examples = dataset["validation"].select(range(min(args.val_size, len(dataset["validation"]))))
else:
    validation_examples = dataset["validation"]
print(f"Loaded {len(validation_examples)} validation examples from SQuAD")

# Prepare features
train_dataset = train_examples.map(
    lambda x: prepare_train_features(x, tokenizer),
    batched=True,
    remove_columns=train_examples.column_names
)

validation_dataset = validation_examples.map(
    lambda x: prepare_validation_features(x, tokenizer),
    batched=True,
    remove_columns=validation_examples.column_names
)

# Convert to torch datasets
train_dataset_final = SQuADDataset({
    'input_ids': train_dataset['input_ids'],
    'attention_mask': train_dataset['attention_mask'],
    'start_positions': train_dataset['start_positions'],
    'end_positions': train_dataset['end_positions']
})

eval_dataset_final = SQuADDataset({
    'input_ids': validation_dataset['input_ids'],
    'attention_mask': validation_dataset['attention_mask'],
    'example_id': validation_dataset['example_id']
})

# Create dataloaders with smart batch sizes
# Smaller batch size to prevent overfitting and better generalization
train_batch_size = args.batch_train_size  # Small batch size for better gradient estimates
eval_batch_size = args.batch_val_size  # Can be larger for evaluation

train_dataloader = DataLoader(train_dataset_final, batch_size=train_batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset_final, batch_size=eval_batch_size)

# Training setup with overfitting prevention
num_epochs = total_epochs  # Total epochs including warmup
warmup_epochs = args.warmup_epochs
learning_rate = args.learning_rate  # Higher LR for LoRA
warmup_ratio = 0.1
weight_decay = 0.01  # L2 regularization

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
total_steps = len(train_dataloader) * num_epochs
warmup_steps = int(warmup_ratio * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Training loop
print(f"\nStarting training for {num_epochs} epochs...")
print(f"  First {warmup_epochs} epochs: Normal training (warmup)")
print(f"  Next {args.epochs} epochs: {'Adversarial training (' + args.perturbation + ')' if args.perturbation else 'Normal training'}")
print(f"Train batch size: {train_batch_size}, Eval batch size: {eval_batch_size}")
print(f"Total training steps: {total_steps}")

training_history = {
    'train_loss': [],
    'train_em': [],
    'train_f1': [],
    'val_em': [],
    'val_f1': [],
    'training_mode': []
}

for epoch in range(num_epochs):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # Determine training mode
    if epoch < warmup_epochs:
        # Warmup phase - always normal training
        use_adversarial = False
        print(f"Training mode: Normal (warmup epoch {epoch + 1}/{warmup_epochs})")
    else:
        # Main training phase - adversarial if perturbation is not 'none'
        use_adversarial = (args.perturbation != None)
        epoch_in_main = epoch - warmup_epochs + 1
        if use_adversarial:
            print(f"Training mode: Adversarial ({args.perturbation}) - epoch {epoch_in_main}/{args.epochs}")
        else:
            print(f"Training mode: Normal - epoch {epoch_in_main}/{args.epochs}")
    
    # Training
    model.train()

    avg_train_loss = train_one_epoch(
        model, train_dataloader, optimizer, scheduler, device, epoch + 1,
        use_adversarial=use_adversarial,
        perturbation_type=args.perturbation,
        epsilon=args.epsilon,
        neftune_alpha=args.neftune_alpha,
        max_grad_norm=1.0
    )
    
    training_history['train_loss'].append(avg_train_loss)
    training_history['training_mode'].append('adversarial' if use_adversarial else 'normal')
    print(f"Average training loss: {avg_train_loss:.4f}")
    
    # Evaluate on training set (subset for efficiency)
    print("Evaluating on training set...")
    train_eval_size = min(500, len(train_examples))
    train_eval_examples = train_examples.select(range(train_eval_size))
    train_eval_dataset = train_eval_examples.map(
        lambda x: prepare_validation_features(x, tokenizer),
        batched=True,
        remove_columns=train_eval_examples.column_names
    )
    train_eval_dataset_final = SQuADDataset({
        'input_ids': train_eval_dataset['input_ids'],
        'attention_mask': train_eval_dataset['attention_mask'],
        'example_id': train_eval_dataset['example_id']
    })
    train_eval_dataloader = DataLoader(train_eval_dataset_final, batch_size=eval_batch_size)
    
    train_metrics = evaluate_model(model, train_eval_dataloader, train_eval_dataset, train_eval_examples)
    training_history['train_em'].append(train_metrics['exact_match'])
    training_history['train_f1'].append(train_metrics['f1'])
    print(f"Train EM: {train_metrics['exact_match']:.2f}%, Train F1: {train_metrics['f1']:.2f}%")
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_metrics = evaluate_model(model, eval_dataloader, validation_dataset, validation_examples)
    training_history['val_em'].append(val_metrics['exact_match'])
    training_history['val_f1'].append(val_metrics['f1'])
    print(f"Val EM: {val_metrics['exact_match']:.2f}%, Val F1: {val_metrics['f1']:.2f}%")

# Print final results
print("\n" + "="*50)
print("=== Training Complete ===")
print("="*50)
print("\nTraining Summary:")
print(f"  Model: {model_full_name}")
print(f"  LoRA rank: {args.lora_rank}")
print(f"  Total epochs: {num_epochs}")
print(f"  Warmup epochs (normal): {warmup_epochs}")
print(f"  Main epochs ({'adversarial ' + args.perturbation if args.perturbation != None else 'vanila'}): {args.epochs}")

print("\nFinal Results:")
print(f"Train - EM: {training_history['train_em'][-1]:.2f}%, F1: {training_history['train_f1'][-1]:.2f}%")
print(f"Val - EM: {training_history['val_em'][-1]:.2f}%, F1: {training_history['val_f1'][-1]:.2f}%")

print("\nDetailed Training History:")
for epoch in range(num_epochs):
    phase = "warmup" if epoch < warmup_epochs else "main"
    print(f"\nEpoch {epoch + 1} ({phase} - {training_history['training_mode'][epoch]} training):")
    print(f"  Loss: {training_history['train_loss'][epoch]:.4f}")
    print(f"  Train - EM: {training_history['train_em'][epoch]:.2f}%, F1: {training_history['train_f1'][epoch]:.2f}%")
    print(f"  Val   - EM: {training_history['val_em'][epoch]:.2f}%, F1: {training_history['val_f1'][epoch]:.2f}%")

if args.perturbation:
    print(f"\nAdversarial Training Details:")
    print(f"  Method: {args.perturbation}")
    if args.perturbation == 'sdbn':
        print(f"  Epsilon: {args.epsilon}")
    elif args.perturbation == 'neftune':
        print(f"  Alpha: {args.neftune_alpha}")
    print(f"  Applied for last {args.epochs} epochs (epochs {warmup_epochs + 1}-{num_epochs})")
    
# Save the model
# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Save training history (EM, F1, loss) with seed number
history_path = os.path.join(output_dir, f"metrics_seed_{args.seed}.json")
with open(history_path, "w") as f:
    json.dump(training_history, f, indent=2)

# Save the model using utils_gen
utils_gen.save_lora_gen_model(model, os.path.join(output_dir, f"model_{args.seed}.pt"))
