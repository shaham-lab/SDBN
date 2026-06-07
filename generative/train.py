# generative.py - SDBN-P (pre-loaded adversarial variants) for decoder-only models
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
import re
import json
import argparse
import warnings
from typing import List, Dict, Tuple
from collections import Counter
import string
import random

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings("ignore")



# ----------------------------
# Models registry
# ----------------------------
models_dict = {
    "llama": "meta-llama/Llama-3.2-1B",
    "llama-7b": "meta-llama/Llama-2-7b-hf",
    "qwen": "Qwen/Qwen2.5-7B",
}

def is_decoder_only_model(model_name: str) -> bool:
    return True  # All supported models are decoder-only (LLaMA/Qwen)

# ----------------------------
# Small utilities
# ----------------------------
def set_random_seed(seed: int):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def resolve_output_dir(args) -> str:
    rel = os.path.join(
        "results", "SDBN-P", args.dataset, args.model_name,
        f"trainset_size_{args.train_size if args.train_size else 'full'}",
        args.peft_method,
        f"rank_{args.lora_rank}",
    )
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        return args.output_dir
    root = args.root if args.root else os.getcwd()
    out = os.path.join(root, rel)
    os.makedirs(out, exist_ok=True)
    return out

def save_lora_gen_model(model, save_path):
    """Save LoRA model (decoder-only: LLaMA/Qwen)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    from peft import get_peft_model_state_dict
    lora_state = get_peft_model_state_dict(model)
    torch.save({"lora": lora_state, "model_type": "decoder_only"}, save_path)
    print(f"[✓] Model saved to {save_path} (LoRA only, {len(lora_state)} tensors)")
    return save_path

def load_lora_gen_model(model_path, model_name, lora_rank=4, device='cuda:0'):
    """Load LoRA model for decoder-only models (LLaMA/Qwen)."""
    from peft import set_peft_model_state_dict

    print(f"Loading model from: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _mn = model_name.lower()
    _dtype = torch.bfloat16 if ("7b" in _mn or "8b" in _mn or "qwen" in _mn) else None
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        torch_dtype=_dtype,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        modules_to_save=[],
    )
    model = get_peft_model(base_model, lora_config)

    if "lora" in state_dict:
        set_peft_model_state_dict(model, state_dict["lora"])
        print(f"[✓] Loaded LoRA weights: {len(state_dict['lora'])} tensors")
    else:
        print("[⚠] No LoRA state found in saved model")

    model = model.to(device)
    model.eval()
    return model


def _get_core_seq2seq_model(model):
    core = model
    if hasattr(core, 'base_model'):
        core = core.base_model
    if hasattr(core, 'model'):
        core = core.model
    return core

# ----------------------------
# Dataset container
# ----------------------------
class GenericDataset(Dataset):
    def __init__(self, encodings: Dict[str, List]):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        item = {}
        for k, v in self.encodings.items():
            if k in ["input_ids", "attention_mask", "labels"]:
                item[k] = torch.tensor(v[idx], dtype=torch.long)
            else:
                item[k] = v[idx]
        return item

class SQuADDataset(GenericDataset):
    pass

class TweetQADataset(GenericDataset):
    pass

# ----------------------------
# Dataset loaders
# ----------------------------
def load_squad():
    """Load SQuAD dataset with train/validation splits"""
    ds = load_dataset("squad")
    train_data = ds["train"]
    val_data = ds["validation"]
    return train_data, val_data

def load_tweetqa():
    """Load TweetQA dataset with train/validation/test splits from parquet"""
    data_files = {
        "train": "https://huggingface.co/datasets/ucsbnlp/tweet_qa/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        "validation": "https://huggingface.co/datasets/ucsbnlp/tweet_qa/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
        "test": "https://huggingface.co/datasets/ucsbnlp/tweet_qa/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
    }
    ds = load_dataset("parquet", data_files=data_files)
    train_data = ds["train"]
    val_data = ds["validation"]
    test_data = ds["test"]  # Store test set for later evaluation
    return train_data, val_data, test_data

# ----------------------------
# Format helpers
# ----------------------------
# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_squad_for_decoder(examples, tokenizer, max_source_len=384, max_target_len=64):
    """
    Preprocess SQuAD for decoder-only training (LLaMA).
    
    *** CRITICAL FIX ***: Remove trailing space from prompt to ensure
    proper alignment between prompt boundary and first answer token.
    """
    input_ids, attention_masks, labels, example_ids = [], [], [], []
    print(f"Processing {len(examples)} SQuAD examples for decoder-only...")
    valid = 0
    
    for idx, ex in enumerate(examples):
        context = (ex.get("context") or "").strip()
        question = (ex.get("question") or "").strip()
        answers = ex.get("answers") or {}
        
        if not context or not question:
            continue
        
        # Get first answer text
        answer_texts = answers.get("text", [])
        if not answer_texts or not answer_texts[0]:
            continue
        answer = answer_texts[0].strip()
        
        # *** FIX: Remove trailing space from prompt ***
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer: "  # No trailing space!
        full_text = f"{prompt}{answer}{tokenizer.eos_token}"
        
        # Tokenize full text (BOS is automatically added at position 0)
        encoded = tokenizer(
            full_text,
            max_length=max_source_len + max_target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize prompt WITHOUT trailing space to get true boundary.
        # The trailing space in "Answer: " merges with the first answer token
        # in the full sequence (e.g. "▁Denver"), so counting it as a prompt
        # token shifts the boundary by 1 and masks the first answer token.
        prompt_encoded = tokenizer(
            prompt.rstrip(' '),
            max_length=max_source_len + max_target_len,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        prompt_len = prompt_encoded["input_ids"].shape[1]
        
        # Create labels: -100 for prompt tokens, actual tokens for answer
        inp_ids = encoded["input_ids"].squeeze().tolist()
        lab = [-100] * len(inp_ids)
        
        # Unmask answer tokens (from prompt_len onwards)
        for i in range(prompt_len, len(inp_ids)):
            if inp_ids[i] != tokenizer.pad_token_id:
                lab[i] = inp_ids[i]
        
        # Count answer tokens for validation
        answer_token_count = sum(1 for l in lab if l != -100)
        
        # *** NEW: Verify we're training on the correct answer ***
        if answer_token_count > 0:
            # Extract what we're actually training on
            trained_tokens = [inp_ids[i] for i in range(len(lab)) if lab[i] != -100]
            trained_text = tokenizer.decode(trained_tokens).strip()
            
            # Check if it matches the expected answer
            if trained_text != answer:
                if idx < 10:  # Only print first 10 warnings
                    print(f"\nExample {idx}:")
                    print(f"  Expected answer: '{answer}'")
                    print(f"  Training on: '{trained_text}'")
                    print(f"  Answer tokens (trained): {answer_token_count}")
                    print(f"  ⚠ WARNING: Mismatch detected!")
        
        # Skip if no answer tokens
        if answer_token_count == 0:
            continue
        
        input_ids.append(inp_ids)
        attention_masks.append(encoded["attention_mask"].squeeze().tolist())
        labels.append(lab)
        example_ids.append(valid)
        valid += 1
        
        # Debug output for first 2 examples
        if idx < 2:
            print(f"\nExample {idx}:")
            print(f"  Context: {context[:80]}...")
            print(f"  Question: {question}")
            print(f"  Answer: {answer}")
            print(f"  Answer tokens (trained): {answer_token_count}")
            if prompt_len < len(inp_ids) and inp_ids[prompt_len] != tokenizer.pad_token_id:
                first_token = tokenizer.decode([inp_ids[prompt_len]])
                print(f"  First trained token at pos {prompt_len}: '{first_token}'")
    
    print(f"SQuAD: created {valid} train instances")
    return {"input_ids": input_ids, "attention_mask": attention_masks,
            "labels": labels, "example_id": example_ids}

def preprocess_tweetqa_for_decoder(examples, tokenizer, max_source_len=512, max_target_len=64):
    input_ids, attention_masks, labels, example_ids = [], [], [], []
    print(f"Processing {len(examples)} TweetQA examples for decoder-only...")
    valid = 0
    
    for idx, ex in enumerate(examples):
        tweet = (ex.get("Tweet") or "").strip()
        question = (ex.get("Question") or "").strip()
        answers = ex.get("Answer") or []
        
        if not tweet or not question:
            continue
        if not answers or (isinstance(answers, list) and len(answers) == 0):
            continue
        answer = answers[0].strip() if isinstance(answers, list) else str(answers).strip()
        
        prompt = f"Tweet: {tweet}\nQuestion: {question}\nAnswer: "
        full_text = f"{prompt}{answer}{tokenizer.eos_token}"
        
        encoded = tokenizer(
            full_text,
            max_length=max_source_len + max_target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        inp_ids = encoded["input_ids"].squeeze().tolist()
        attn = encoded["attention_mask"].squeeze().tolist()
        
        # Find prompt boundary: account for BOS prepended by tokenizer
        prompt_only = tokenizer(prompt, add_special_tokens=False,
                                truncation=False, return_tensors="pt")
        prompt_ids = prompt_only["input_ids"].squeeze().tolist()
        has_bos = len(inp_ids) > 0 and inp_ids[0] == tokenizer.bos_token_id
        prompt_len = (1 if has_bos else 0) + len(prompt_ids)
        
        # Build labels: use attention_mask (not pad_token_id) so EOS is kept
        # This teaches the model to stop generating
        lab = [-100] * len(inp_ids)
        for i in range(prompt_len, len(inp_ids)):
            if attn[i] == 1:  # real token, not padding
                lab[i] = inp_ids[i]
        
        answer_token_count = sum(1 for l in lab if l != -100)
        if answer_token_count == 0:
            continue
        
        input_ids.append(inp_ids)
        attention_masks.append(attn)
        labels.append(lab)
        example_ids.append(valid)
        valid += 1
        
        if idx < 2:
            print(f"\nExample {idx}:")
            print(f"  Tweet: {tweet[:80]}...")
            print(f"  Question: {question}")
            print(f"  Answer: {answer}")
            print(f"  Answer tokens (trained): {answer_token_count}")
    
    print(f"TweetQA: created {valid} train instances")
    return {"input_ids": input_ids, "attention_mask": attention_masks,
            "labels": labels, "example_id": example_ids}

# ----------------------------
# Eval helpers
# ----------------------------
from contextlib import contextmanager
@contextmanager
def eval_mode(m):
    was_training = m.training
    m.eval()
    try:
        yield
    finally:
        if was_training:
            m.train()

def normalize_answer(s: str) -> str:
    """Normalize answer text for SQuAD evaluation"""
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

def compute_squad_em(prediction: str, ground_truth: str) -> int:
    """Compute exact match for SQuAD"""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_squad_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score for SQuAD"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

@torch.no_grad()
def evaluate_squad_em_f1(model, tokenizer, examples, max_src_len=384, max_gen_len=64,
                        num_samples=200, debug=False, dataset_name="SQuAD"):
    """
    Evaluate QA datasets (SQuAD, AdversarialQA) using EM and F1 metrics for decoder-only models.
    Format: Context: <context>\nQuestion: <question>\nAnswer:
    """
    device = next(model.parameters()).device
    K = min(num_samples, len(examples))
    subset = examples[:K] if isinstance(examples, list) else [examples[i] for i in range(K)]
    
    em_scores = []
    f1_scores = []
    
    with eval_mode(model):
        for i, ex in enumerate(tqdm(subset, desc=f"Evaluating {dataset_name} (EM/F1)")):
            context = (ex.get("context") or "").strip()
            question = (ex.get("question") or "").strip()
            answers = ex.get("answers") or {}
            
            if not context or not question:
                continue
            
            # Get ground truth answer
            answer_texts = answers.get("text", [])
            if not answer_texts:
                continue
            ground_truth = answer_texts[0].strip()
            
            # Generate prediction
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer: "
            inp = tokenizer(prompt, max_length=max_src_len, truncation=True,
                           return_tensors="pt").to(device)
            
            out_ids = model.generate(
                **inp,
                max_new_tokens=max_gen_len,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Slice off prompt from generated sequence
            prompt_len = inp["input_ids"].shape[1]
            pred_ids = out_ids[0][prompt_len:]
            prediction = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            
            # Stop at newline or common continuation markers
            for stop_marker in ['\n', 'Context:', 'Question:', 'Explanation:', 'Note:']:
                if stop_marker in prediction:
                    prediction = prediction.split(stop_marker)[0].strip()
                    break
            
            # Compute metrics
            em = compute_squad_em(prediction, ground_truth)
            f1 = compute_squad_f1(prediction, ground_truth)
            
            em_scores.append(em)
            f1_scores.append(f1)
            
            if debug and i < 5:
                print(f"\n{dataset_name} example {i+1}:")
                print(f"  Context: {context[:100]}...")
                print(f"  Question: {question}")
                print(f"  Ground truth: {ground_truth}")
                print(f"  Prediction: {prediction}")
                print(f"  EM: {em}, F1: {f1:.2f}")
                print("-" * 60)
    
    if len(em_scores) == 0:
        return 0.0, 0.0
    
    em_score = 100.0 * sum(em_scores) / len(em_scores)
    f1_score = 100.0 * sum(f1_scores) / len(f1_scores)
    
    print(f"\n{dataset_name} Results: EM={em_score:.2f}%, F1={f1_score:.2f}% ({len(em_scores)} examples)")
    
    # Clear cache after evaluation
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return float(em_score), float(f1_score)

@torch.no_grad()
def evaluate_tweetqa_em_f1(model, tokenizer, examples, max_src_len=384, max_gen_len=64,
                           num_samples=200, debug=False):
    """
    Evaluate TweetQA using EM and F1 metrics for decoder-only models.
    TweetQA structure: Tweet, Question, Answer (list)
    Format: Tweet: <tweet>\nQuestion: <question>\nAnswer:
    """
    device = next(model.parameters()).device
    K = min(num_samples, len(examples))
    subset = examples[:K] if isinstance(examples, list) else [examples[i] for i in range(K)]
    
    em_scores = []
    f1_scores = []
    
    # Switch to left-padding for generation (decoder-only models need this for proper attention)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    with eval_mode(model):
        for i, ex in enumerate(tqdm(subset, desc="Evaluating TweetQA (EM/F1)")):
            tweet = (ex.get("Tweet") or "").strip()
            question = (ex.get("Question") or "").strip()
            answers = ex.get("Answer") or []
            
            if not tweet or not question:
                continue
            
            # Get ground truth answers (Answer is a list — may have multiple valid answers)
            if not answers or (isinstance(answers, list) and len(answers) == 0):
                continue
            ref_list = [a.strip() for a in answers] if isinstance(answers, list) else [str(answers).strip()]
            ref_list = [a for a in ref_list if a]
            if not ref_list:
                continue

            # Generate prediction
            prompt = f"Tweet: {tweet}\nQuestion: {question}\nAnswer: "
            inp = tokenizer(prompt, max_length=max_src_len, truncation=True,
                           return_tensors="pt").to(device)

            out_ids = model.generate(
                        **inp,
                        max_new_tokens=32,
                        num_beams=1,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.3,
                    )

            # Slice off prompt from generated sequence
            prompt_len = inp["input_ids"].shape[1]
            pred_ids = out_ids[0][prompt_len:]
            prediction = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

            # Stop at newline or common continuation markers
            for stop_marker in ['\n', 'Tweet:', 'Question:', 'Explanation:', 'Note:']:
                if stop_marker in prediction:
                    prediction = prediction.split(stop_marker)[0].strip()
                    break

            # Compute metrics: take max over all reference answers (standard SQuAD protocol)
            em = max(compute_squad_em(prediction, ref) for ref in ref_list)
            f1 = max(compute_squad_f1(prediction, ref) for ref in ref_list)
            
            em_scores.append(em)
            f1_scores.append(f1)
            
            if debug and i < 5:
                print(f"\nTweetQA example {i+1}:")
                print(f"  Tweet: {tweet[:100]}...")
                print(f"  Question: {question}")
                print(f"  Ground truth: {ref_list}")
                print(f"  Prediction: {prediction}")
                print(f"  EM: {em}, F1: {f1:.2f}")
                print("-" * 60)
    
    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    
    if len(em_scores) == 0:
        return 0.0, 0.0
    
    em_score = 100.0 * sum(em_scores) / len(em_scores)
    f1_score = 100.0 * sum(f1_scores) / len(f1_scores)
    
    print(f"\nTweetQA Results: EM={em_score:.2f}%, F1={f1_score:.2f}% ({len(em_scores)} examples)")
    
    # Clear cache after evaluation
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return float(em_score), float(f1_score)

@torch.no_grad()
# ----------------------------
# Helper function for SDBN-PL/SDBN-PS (pre-loaded adversarial variants)
# ----------------------------
def load_adversarial_csv(output_dir, args, epoch_num=None):
    """Load pre-generated adversarial examples from CSV file(s).
    
    For sdbn-p with multi_csv=False: loads from sdbn-p_data/{dataset}/trainset_{train_size}/trainset_seed{seed}_sdbn_p.csv
    For sdbn-p with multi_csv=True: loads from sdbn-p_data/{dataset}/trainset_{train_size}/trainset_seed{seed}_sdbn_p_e{epoch}.csv
    For sdbn-p: loads from output_dir/trainset_seed{seed}_sdbn_p.csv
    
    Args:
        output_dir: Output directory
        args: Arguments object
        epoch_num: Epoch number (used when multi_csv=True to select specific CSV)
    
    Returns dict mapping example_id -> list of (prompt, answer) tuples.
    The CSV has columns: input, answer, input_adv1, ..., input_adv5
    We skip 'input' column and only load adversarial variants.
    """
    import csv
    
    # Use appropriate CSV filename based on perturbation type
    suffix = args.perturbation.split('-')[1]  # 'pl' or 'ps'
    
    # Handle multi-CSV mode for SDBN-PL with epoch-specific variants
    # If epoch_num is provided: combine it into filename (multi_csv mode)
    # If epoch_num is None: use regular filename (single file mode)
    if args.perturbation == 'sdbn-p' and epoch_num is not None:
        csv_filename = f"trainset_seed{args.seed}_sdbn_p_e{epoch_num}.csv"
    else:
        csv_filename = f"trainset_seed{args.seed}_sdbn_{suffix}.csv"
    
    if args.perturbation == 'sdbn-p':
        # sdbn-p_data/{dataset}/trainset_{train_size}/ relative to --root or cwd
        if args.resource_dir:
            base = args.resource_dir
        else:
            root = args.root if args.root else os.getcwd()
            base = os.path.join(root, 'sdbn-p_data', args.dataset)
        train_size_str = args.train_size if args.train_size else 'full'
        csv_path = os.path.join(base, f"trainset_{train_size_str}", csv_filename)
    else:
        # sdbn-p: keep loading from output_dir as before
        csv_path = os.path.join(output_dir, csv_filename)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"SDBN-PL mode requires pre-generated CSV file: {csv_path}\n"
            f"Please generate this file first using the appropriate script."
        )
        exit(1)
    
    if epoch_num is not None:
        print(f"\nLoading epoch-specific adversarial examples (e{epoch_num}) from: {csv_path}")
    else:
        print(f"\nLoading pre-generated adversarial examples from: {csv_path}")
    
    adversarial_variants = {}
    rows_processed = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for idx, row in enumerate(reader):
            rows_processed += 1
            
            # Debug: Print first row to verify column names
            if idx == 0:
                print(f"[DEBUG] CSV columns: {list(row.keys())}")
                print(f"[DEBUG] First row keys present: {[k for k in row.keys() if row[k]]}")
            
            # Extract clean input and answer
            clean_input = row.get('input', '').strip()
            answer = row.get('answer', '').strip()
            
            # Build variants list: [clean_input, adv1, adv2, ...]
            # Auto-detect number of adversarial variants from CSV columns.
            variants = []
            if clean_input:
                variants.append((clean_input, answer))  # First element is clean input

            # Extract all input_advN columns present in this row
            i = 1
            while True:
                adv_key = f'input_adv{i}'
                if adv_key not in row:
                    break
                adv_val = row[adv_key].strip()
                if adv_val:
                    variants.append((adv_val, answer))
                    if idx == 0:  # Debug first row
                        print(f"[DEBUG] Found variant {adv_key}: {adv_val[:50]}...")
                i += 1

            if variants:
                adversarial_variants[idx] = variants
                if idx < 3:  # Debug first few examples
                    print(f"[DEBUG] Example {idx}: {len(variants)} variants loaded")
    
    print(f"[DEBUG] Processed {rows_processed} rows from CSV")
    print(f"Loaded {len(adversarial_variants)} examples with adversarial variants")
    if len(adversarial_variants) > 0:
        print(f"[DEBUG] Sample IDs with variants: {list(adversarial_variants.keys())[:10]}")
    return adversarial_variants
# for tweetqa AND squad — unified, fair, with clean_prob support
def select_max_loss_variant(model, tokenizer, variants, device, is_decoder_only=True,
                            max_source_len=512, max_target_len=128, dataset='squad',
                            greedy_prob=1.0, temperature=1.0, clean_prob=0.0):
    """Select a variant using mixed greedy/sampling strategy.

    FAIRNESS GUARANTEE: label masking and prompt-boundary detection mirror
    the *exact* logic used by each dataset's preprocessing function so that
    the loss landscape seen during selection is identical to the one seen
    during the actual training forward pass.

    Supported datasets:
        squad    – preprocess_squad_for_decoder style
        tweetqa  – preprocess_tweetqa_for_decoder style

    Args:
        model:          The model (switched to eval internally, restored to train)
        tokenizer:      Tokenizer
        variants:       List of (prompt_text, answer_text) tuples.
                        variants[0] is always the clean input.
        device:         torch device
        is_decoder_only: Whether model is decoder-only
        max_source_len: --max_length
        max_target_len: --max_target_length
        dataset:        Dataset name string
        greedy_prob:    Probability of argmax selection (1.0 = always greedy)
        temperature:    Softmax temperature when sampling
        clean_prob:     Probability of forcing variant 0 (clean input).
                        Checked FIRST, before greedy/sampling.
                        Use 0.0 for SQuAD (already works), 0.3 for TweetQA.

    Returns:
        (encoded_dict, variant_index)
        encoded_dict has keys 'input_ids', 'attention_mask', 'labels' on device
    """
    model.eval()

    max_len = max_source_len + max_target_len
    all_losses = []
    all_encoded = []

    with torch.no_grad():
        for var_idx, (prompt_text, answer_text) in enumerate(variants):

            # ----------------------------------------------------------
            # 1. Build full_text and determine prompt connector
            # ----------------------------------------------------------
            if not is_decoder_only:
                # seq2seq (T5/BART) – source and target are separate
                encoded = tokenizer(prompt_text, max_length=max_source_len,
                                    truncation=True, return_tensors="pt")
                target_encoded = tokenizer(answer_text, max_length=max_target_len,
                                           truncation=True, return_tensors="pt")
                labels = target_encoded['input_ids']

            else:
                # ---- decoder-only: prompt construction ----
                # squad/tweetqa: "Context: …\nQuestion: …" or "Tweet: …\nQuestion: …"
                prompt_with_answer = f"{prompt_text}\nAnswer: {answer_text}{tokenizer.eos_token}"

                # ---- tokenise full sequence ----
                encoded = tokenizer(
                    prompt_with_answer,
                    max_length=max_len,
                    truncation=True,
                    return_tensors="pt",
                )
                inp_ids = encoded['input_ids']          # [1, seq_len]
                attn    = encoded['attention_mask']      # [1, seq_len]

                # ----------------------------------------------------------
                # 2. Detect prompt boundary
                #    We replicate the method used by each dataset's
                #    preprocess_*_for_decoder function.
                # ----------------------------------------------------------
                if dataset == 'tweetqa':
                    # === MATCHES preprocess_tweetqa_for_decoder EXACTLY ===
                    reconstructed_prompt = f"{prompt_text}\nAnswer: "
                    prompt_only = tokenizer(
                        reconstructed_prompt,
                        add_special_tokens=False,
                        truncation=False,
                        return_tensors="pt",
                    )
                    prompt_ids = prompt_only['input_ids'].squeeze()
                    has_bos = (inp_ids.shape[1] > 0 and
                               inp_ids[0, 0].item() == tokenizer.bos_token_id)
                    prompt_len = (1 if has_bos else 0) + len(prompt_ids)

                else:  # squad
                    # === MATCHES preprocess_squad_for_decoder EXACTLY ===
                    reconstructed_prompt = f"{prompt_text}\nAnswer: "
                    prompt_enc = tokenizer(
                        reconstructed_prompt.rstrip(' '),
                        add_special_tokens=True,
                        truncation=True,
                        max_length=max_len,
                        return_tensors="pt",
                    )
                    prompt_len = prompt_enc['input_ids'].shape[1]

                # ----------------------------------------------------------
                # 3. Build labels – match each dataset's masking convention
                # ----------------------------------------------------------
                labels = torch.full_like(inp_ids, -100)  # start all masked

                if dataset == 'tweetqa':
                    # === MATCHES preprocess_tweetqa_for_decoder ===
                    # Uses attention_mask: keeps EOS in labels
                    for i in range(prompt_len, inp_ids.shape[1]):
                        if attn[0, i].item() == 1:
                            labels[0, i] = inp_ids[0, i]

                else:  # squad
                    # === MATCHES preprocess_squad_for_decoder ===
                    # Uses pad_token_id check: masks EOS (since pad=eos)
                    for i in range(prompt_len, inp_ids.shape[1]):
                        if inp_ids[0, i].item() != tokenizer.pad_token_id:
                            labels[0, i] = inp_ids[0, i]

            # ----------------------------------------------------------
            # 4. Skip if no trainable tokens
            # ----------------------------------------------------------
            if (labels != -100).sum().item() == 0:
                continue

            # ----------------------------------------------------------
            # 5. Forward pass → loss
            # ----------------------------------------------------------
            outputs = model(
                input_ids=encoded['input_ids'].to(device),
                attention_mask=encoded['attention_mask'].to(device),
                labels=labels.to(device),
                return_dict=True,
            )

            all_losses.append(outputs.loss.item())
            all_encoded.append({
                'input_ids':      encoded['input_ids'].to(device),
                'attention_mask': encoded['attention_mask'].to(device),
                'labels':         labels.to(device),
            })

    # ----------------------------------------------------------
    # 6. Selection: clean override → greedy → sampling
    # ----------------------------------------------------------
    if not all_losses:
        raise ValueError("select_max_loss_variant: no valid variants (all had 0 trainable tokens)")

    losses_tensor = torch.tensor(all_losses)

    # CHANGED: clean_prob check FIRST — guarantees a fixed fraction
    # of training examples use the clean (unperturbed) input.
    # variant 0 is always clean (first entry in CSV = original input).
    if clean_prob > 0 and random.random() < clean_prob:
        best_idx = 0
    elif random.random() < greedy_prob:
        best_idx = losses_tensor.argmax().item()
    else:
        probs = torch.softmax(losses_tensor / temperature, dim=0)
        best_idx = torch.multinomial(probs, 1).item()

    model.train()
    return all_encoded[best_idx], best_idx


# ----------------------------
# Training step
# ----------------------------
def train_one_epoch(model, train_dataloader, optimizer, scheduler, device, epoch, max_grad_norm=1.0,
                   use_adversarial=False, perturbation_type='sdbn-p',
                   adversarial_variants=None, tokenizer=None,
                   max_source_len=512, max_target_len=128, dataset='squad', mem_peak_scope: str = 'epoch'):
    # Clear GPU cache before training epoch
        
    model.train()
    torch.cuda.empty_cache()

    # Track peak GPU memory.
    # mem_peak_scope:
    # - 'epoch': peak across the entire training epoch loop (training only).
    # - 'fwd_bwd': peak only within each forward+backward block (excludes variant search / attack generation).
    # - 'none': disable measurement.
    peak_alloc_gib = 0.0
    peak_reserved_gib = 0.0
    if device.type == 'cuda' and mem_peak_scope == 'epoch':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    def _maybe_reset_peak_before_fwd():
        if device.type == 'cuda' and mem_peak_scope == 'fwd_bwd':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

    def _update_peak_from_cuda_stats():
        nonlocal peak_alloc_gib, peak_reserved_gib
        if device.type != 'cuda' or mem_peak_scope == 'none':
            return
        torch.cuda.synchronize(device)
        gib = 1024 ** 3
        peak_alloc_gib = max(peak_alloc_gib, torch.cuda.max_memory_allocated(device) / gib)
        peak_reserved_gib = max(peak_reserved_gib, torch.cuda.max_memory_reserved(device) / gib)
    
    total_loss = 0.0
    valid_batches = 0
    epoch_variant_choices = []  # tracks selected variant index per example (sdbn-p only)
    training_mode = "adversarial" if use_adversarial else "normal"
    if use_adversarial:
        training_mode += f" ({perturbation_type})"
    
    pb = tqdm(train_dataloader, desc=f"Training (epoch {epoch}) - {training_mode}")
    for batch_idx, batch in enumerate(pb):
        torch.cuda.empty_cache()
        example_ids = batch.pop("example_id", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # SDBN-PL/SDBN-PS mode: Select best adversarial variant for each example
        if use_adversarial and perturbation_type in 'sdbn-p':
            if adversarial_variants is None or tokenizer is None:
                raise ValueError("SDBN-PL/SDBN-PS requires adversarial_variants and tokenizer")
            
            # Debug: Check if any variants are available at all
            if batch_idx == 0:
                print(f"\n[SDBN-PL DEBUG] Total variants available: {len(adversarial_variants)}")
                if len(adversarial_variants) > 0:
                    print(f"[SDBN-PL DEBUG] Variant IDs (first 10): {list(adversarial_variants.keys())[:10]}")
            
            # Check if decoder-only model
            core = _get_core_seq2seq_model(model)
            is_decoder_only = not hasattr(core, 'encoder')
            
            # Build batch by selecting max-loss variant for each example
            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []
            
            # Track adversarial vs fallback
            adv_count = 0
            fallback_count = 0
            
            for i in range(batch['input_ids'].shape[0]):
                # Extract example ID (handle both tensor and raw values)
                if example_ids is not None:
                    ex_id = example_ids[i].item() if torch.is_tensor(example_ids[i]) else example_ids[i]
                    
                    # Debug first batch
                    if batch_idx == 0 and i == 0:
                        print(f"[SDBN-PL DEBUG] First example ID: {ex_id} (type: {type(ex_id)})")
                        print(f"[SDBN-PL DEBUG] ID in variants dict? {ex_id in adversarial_variants}")
                    
                    if ex_id in adversarial_variants:
                        variants = adversarial_variants[ex_id]
                        selected, var_idx = select_max_loss_variant(
                            model, tokenizer, variants, device, is_decoder_only,
                            max_source_len=max_source_len, max_target_len=max_target_len,
                            dataset=dataset
                        )

                        if selected is not None:
                            batch_input_ids.append(selected['input_ids'])
                            batch_attention_mask.append(selected['attention_mask'])
                            batch_labels.append(selected['labels'])
                            epoch_variant_choices.append(var_idx)
                            adv_count += 1
                        else:
                            # Fallback to original batch item
                            batch_input_ids.append(batch['input_ids'][i:i+1])
                            batch_attention_mask.append(batch['attention_mask'][i:i+1])
                            batch_labels.append(batch['labels'][i:i+1])
                            fallback_count += 1
                    else:
                        # Debug why fallback happens
                        if batch_idx == 0 and i < 3:
                            print(f"[SDBN-PL DEBUG] Example {i} (ID={ex_id}): NOT in variants, using fallback")
                        
                        # Fallback to original batch item if no variants available
                        batch_input_ids.append(batch['input_ids'][i:i+1])
                        batch_attention_mask.append(batch['attention_mask'][i:i+1])
                        batch_labels.append(batch['labels'][i:i+1])
                        fallback_count += 1
                else:
                    # No example IDs available
                    if batch_idx == 0:
                        print(f"[{perturbation_type.upper()} DEBUG] No example_ids in batch!")
                    batch_input_ids.append(batch['input_ids'][i:i+1])
                    batch_attention_mask.append(batch['attention_mask'][i:i+1])
                    batch_labels.append(batch['labels'][i:i+1])
                    fallback_count += 1
            
            # Print statistics for every batch
            print(f"[SDBN-PL batch {batch_idx}] Adversarial: {adv_count}/{adv_count+fallback_count}, Fallback: {fallback_count}")
            
            max_len = max_source_len + max_target_len
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            
            # Pad all tensors to max_len
            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []
            
            for inp_ids, attn_mask, labs in zip(batch_input_ids, batch_attention_mask, batch_labels):
                seq_len = inp_ids.shape[1]
                if seq_len < max_len:
                    # Pad on the right
                    pad_len = max_len - seq_len
                    inp_ids = torch.cat([inp_ids, torch.full((1, pad_len), pad_token_id, dtype=torch.long, device=device)], dim=1)
                    attn_mask = torch.cat([attn_mask, torch.zeros((1, pad_len), dtype=torch.long, device=device)], dim=1)
                    labs = torch.cat([labs, torch.full((1, pad_len), -100, dtype=torch.long, device=device)], dim=1)
                
                padded_input_ids.append(inp_ids)
                padded_attention_mask.append(attn_mask)
                padded_labels.append(labs)
            
            # Concatenate into batch tensors
            batch['input_ids'] = torch.cat(padded_input_ids, dim=0)
            batch['attention_mask'] = torch.cat(padded_attention_mask, dim=0)
            batch['labels'] = torch.cat(padded_labels, dim=0)
            
            # Standard forward pass (no adversarial perturbation needed)
            optimizer.zero_grad()
            _maybe_reset_peak_before_fwd()
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                return_dict=True
            )
            
        else:
            optimizer.zero_grad()
            _maybe_reset_peak_before_fwd()
            outputs = model(**batch)
        
        loss = outputs.loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[warn] skip batch {batch_idx} invalid loss {loss}")
            continue
        
        total_loss += loss.item()
        valid_batches += 1
        
        # Single backward pass for all adversarial methods (fair-fight approach)
        loss.backward()

        # In fwd_bwd mode, record peak right after backward.
        if device.type == 'cuda' and mem_peak_scope == 'fwd_bwd':
            _update_peak_from_cuda_stats()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        if device.type == 'cuda' and mem_peak_scope == 'epoch':
            # Sample during epoch; final value is read at end.
            if batch_idx % 50 == 0:
                _update_peak_from_cuda_stats()
        
        if batch_idx % 10 == 0:
            pb.set_postfix({"loss": f"{loss.item():.3f}", "avg": f"{total_loss/max(1, valid_batches):.3f}", "mode": training_mode})
    
    avg = total_loss / max(1, valid_batches)
    print(f"Epoch {epoch} completed: {valid_batches} valid batches, Avg Loss: {avg:.4f} ({training_mode})")
    if epoch_variant_choices:
        from collections import Counter
        counts = Counter(epoch_variant_choices)
        print(f"[Epoch {epoch} SDBN-PL] Variant choices per example (example_idx -> variant_idx):")
        for ex_i, v_idx in enumerate(epoch_variant_choices):
            print(f"  example {ex_i:>4d} -> variant {v_idx}")
        print(f"[Epoch {epoch} SDBN-PL] Distribution: { {k: counts[k] for k in sorted(counts)} }")

    if device.type == 'cuda' and mem_peak_scope != 'none':
        # Final readback.
        _update_peak_from_cuda_stats()

        gib = 1024 ** 3

        # Sanity-check against device capacity (helps catch device-mismatch/summing bugs).
        try:
            props = torch.cuda.get_device_properties(device)
            total_gib = props.total_memory / gib
            if peak_reserved_gib > total_gib * 1.10:
                print(
                    f"[warn][MEMORY] Peak reserved ({peak_reserved_gib:.2f} GiB) exceeds device total ({total_gib:.2f} GiB). "
                    "This usually means you're reading stats from the wrong CUDA device or aggregating multiple GPUs."
                )
        except Exception:
            props = None
            total_gib = None

        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        name = getattr(props, 'name', None) if props is not None else None
        total_str = f"/{total_gib:.2f} GiB" if total_gib is not None else ""
        name_str = f" {name}" if name else ""
        cur_idx = torch.cuda.current_device()
        scope_tag = "FWD+BWD" if mem_peak_scope == 'fwd_bwd' else "EPOCH"
        print(
            f"[MEMORY] Peak GPU memory ({scope_tag}, epoch {epoch}) on cuda:{dev_idx}{name_str}{total_str} "
            f"(current cuda:{cur_idx}): allocated={peak_alloc_gib:.2f} GiB, reserved={peak_reserved_gib:.2f} GiB"
        )
    
    # Clear cache after epoch
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return avg, peak_alloc_gib, peak_reserved_gib

# ----------------------------
# CLI defaults
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Seq2Seq fine-tuning with LoRA (JFLEG/E2E/WikiAuto/PersonaChat)")

    p.add_argument(
        '--perturbation', '-pt',
        type=str,
        choices=['sdbn-p'],
        default='sdbn-p',
        help='Type of adversarial perturbation to use'
    )

    p.add_argument("--warmup_epochs", "-we", type=int, default=1)
    p.add_argument("--epochs", "-e", type=int, default=5)

    p.add_argument("--model_name", type=str, default="llama", choices=list(models_dict.keys()))
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--peft_method", type=str, default="lora", choices=["lora","none"])

    p.add_argument("--dataset", type=str, default="squad", choices=["squad", "tweetqa"])
    p.add_argument("--train_size", "-tr", type=int, default=0)
    p.add_argument("--val_size", "-va", type=int, default=0)
    p.add_argument("--resource_dir", type=str, default=None, help="Directory containing pre-generated adversarial CSV files for SDBN-P")
    p.add_argument('--multi_csv', action='store_true', default=False,
                    help='For SDBN-P: load different CSV files for each epoch starting after warmup (format: trainset_seed{S}_sdbn_p_e{E}.csv where E = epoch number)')

    p.add_argument(
        "--mem_peak_scope",
        type=str,
        default="fwd_bwd",
        choices=["epoch", "fwd_bwd", "none"],
        help=(
            "How to measure peak GPU memory during training. "
            "'epoch' = peak across the training epoch loop; "
            "'fwd_bwd' = peak only within each forward+backward block (excludes variant search / attack generation); "
            "'none' = disable."
        ),
    )
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--max_target_length", type=int, default=32)

    p.add_argument("--batch_train_size", type=int, default=16)
    p.add_argument("--batch_val_size", type=int, default=200)
    p.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0)

    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--fixed_val_size", type=int, default=500)
    p.add_argument("--skip_train_eval", action="store_true", default=False,
                    help="Skip performance evaluation on train set (faster training)")
    p.add_argument("--skip_all_eval", action="store_true", default=False,
                    help="Skip ALL evaluation (train and val) for fastest training")
    p.add_argument("--save_trainset", action="store_true", default=False,
                    help="Save training dataset to CSV in output directory")
    p.add_argument("--not_overwrite", action="store_true", default=False,
                    help="If set, skip training if the final model for this seed already exists in output_dir")
    p.add_argument("--root", default='', action="store", type=str, help="Root directory for data/model storage (default: current directory)")
    return p.parse_args()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("Starting SDBN-P generative fine-tuning with the following configuration:")
    
    args = parse_args()

    # Early exit if --not_overwrite is set and the final model already exists
    if args.not_overwrite:
        # Determine where the model would be saved (use args.output_dir if given, else resolve)
        _check_dir = args.output_dir if args.output_dir else resolve_output_dir(args)
        expected_model_path = os.path.join(_check_dir, f"model_seed_{args.seed}.pt")
        if os.path.isfile(expected_model_path):
            print(f"[not_overwrite] Model already exists at {expected_model_path}. Skipping training.")
            import sys; sys.exit(0)

    set_random_seed(args.seed)

    model_full_name = models_dict[args.model_name]
    output_dir = resolve_output_dir(args)
    
    print(f"Resolved output directory: {output_dir}")
    import sys

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {model_full_name}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Main epochs: {args.epochs}")
    total_epochs = args.warmup_epochs + args.epochs
    print(f"  Total epochs: {total_epochs}")
    print(f"  Perturbation: {args.perturbation}")
    if args.perturbation:
        print(f"  Adversarial training will be used for the last {args.epochs} epochs")
        if args.perturbation == 'sdbn-p':
            if args.multi_csv:
                csv_start = args.warmup_epochs + 1
                csv_end = total_epochs
                print(f"  Multi-CSV mode: ENABLED (epochs {csv_start} to {csv_end}, format: trainset_seed{{S}}_sdbn_p_e{{E}}.csv)")
            else:
                print(f"  Multi-CSV mode: DISABLED (using single CSV: trainset_seed{{S}}_sdbn_p.csv)")
    else:
        print(f"  All {total_epochs} epochs will use normal training")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Train batch size: {args.batch_train_size}")
    print(f"  Max src len: {args.max_length} | Max tgt len: {args.max_target_length}")

    # Load tokenizer and model based on architecture
    tokenizer = AutoTokenizer.from_pretrained(model_full_name, token=os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load appropriate model type
    is_decoder_only = is_decoder_only_model(model_full_name)
    if is_decoder_only:
        # Keep right-padding for training (preprocessing needs it for correct label masking)
        # We'll switch to left-padding only during generation in evaluation functions
        tokenizer.padding_side = 'right'
        print(f"[INFO] Using padding_side='right' for training (will switch to 'left' during generation)")
        
        _mn = model_full_name.lower()
        _dtype = torch.bfloat16 if ("7b" in _mn or "8b" in _mn or "qwen" in _mn) else None
        model = AutoModelForCausalLM.from_pretrained(
            model_full_name,
            token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
            torch_dtype=_dtype,
        )
        print(f"Loaded decoder-only model: {model_full_name} (dtype={_dtype})")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_full_name)
        print(f"Loaded encoder-decoder model: {model_full_name}")

    if args.peft_method == "lora":
        if "bart" in args.model_name:
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        elif "llama" in args.model_name or "qwen" in args.model_name:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:  # T5
            target_modules = ["q","k","v","o","wi","wo"]
        
        # Select task type based on model architecture
        task_type = TaskType.CAUSAL_LM if is_decoder_only else TaskType.SEQ_2_SEQ_LM
        
        lcfg = LoraConfig(
            task_type=task_type,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            modules_to_save=[],
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
    model.to(device)

    # Initialize test_examples for datasets with test sets (currently TweetQA)
    test_examples = None

    print(f"\n{'='*60}\nLoading {args.dataset} dataset...\n{'='*60}")
    if args.dataset == "squad":
        train_examples, val_examples = load_squad()
    elif args.dataset == "tweetqa":
        train_examples, val_examples, test_examples = load_tweetqa()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.train_size > 0 and args.train_size < len(train_examples):
        print(f"Subsampling train: {len(train_examples)} → {args.train_size}")
        train_examples = train_examples.select(range(args.train_size))
    if args.val_size > 0 and args.val_size < len(val_examples):
        print(f"Subsampling val: {len(val_examples)} → {args.val_size}")
        val_examples = val_examples.select(range(args.val_size))
    
    # Subsample test set if needed (TweetQA)
    if test_examples is not None and args.val_size > 0 and args.val_size < len(test_examples):
        print(f"Subsampling test: {len(test_examples)} → {args.val_size}")
        test_examples = test_examples.select(range(args.val_size))

    print(f"Train examples: {len(train_examples)} | Val examples: {len(val_examples)}")

    if hasattr(val_examples, 'select'):
        fixed_val_eval = val_examples.select(range(min(args.fixed_val_size, len(val_examples))))
    else:
        fixed_val_eval = val_examples[:min(args.fixed_val_size, len(val_examples))]

    if hasattr(train_examples, 'shuffle'):
        train_examples = train_examples.shuffle(seed=args.seed)
    if hasattr(val_examples, 'shuffle'):
        fixed_val_eval = val_examples.select(range(min(args.fixed_val_size, len(val_examples))))

    if args.dataset in ["squad"]:
        if not is_decoder_only:
            raise ValueError("SQuAD-style QA datasets currently only support decoder-only models (LLaMA)")
        tr_data = preprocess_squad_for_decoder(
            train_examples, tokenizer,
            max_source_len=args.max_length,
            max_target_len=args.max_target_length
        )
        va_data = preprocess_squad_for_decoder(
            val_examples, tokenizer,
            max_source_len=args.max_length,
            max_target_len=args.max_target_length
        )
        train_ds = SQuADDataset(tr_data)
        val_ds = SQuADDataset(va_data)
    elif args.dataset == "tweetqa":
        if not is_decoder_only:
            raise ValueError("TweetQA currently only supports decoder-only models (LLaMA)")
        tr_data = preprocess_tweetqa_for_decoder(
            train_examples, tokenizer,
            max_source_len=args.max_length,
            max_target_len=args.max_target_length
        )
        va_data = preprocess_tweetqa_for_decoder(
            val_examples, tokenizer,
            max_source_len=args.max_length,
            max_target_len=args.max_target_length
        )
        train_ds = TweetQADataset(tr_data)
        val_ds = TweetQADataset(va_data)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"\nDataset sizes after preprocessing: train={len(train_ds)} | val={len(val_ds)}")

    # Save training dataset to CSV if requested
    if args.save_trainset:
        import csv
        import os
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"trainset_seed{args.seed}.csv")
        
        print(f"\nSaving training dataset to: {csv_path}")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header based on dataset type
            if args.dataset == "squad":
                writer.writerow(["input", "answer"])
                for ex in train_examples:
                    context = (ex.get("context") or "").strip()
                    question = (ex.get("question") or "").strip()
                    input_text = f"Context: {context}\nQuestion: {question}"
                    answers = ex.get("answers") or {}
                    answer_texts = answers.get("text", [])
                    answer = answer_texts[0].strip() if answer_texts else ""
                    writer.writerow([input_text, answer])
            elif args.dataset == "tweetqa":
                writer.writerow(["input", "answer"])
                for ex in train_examples:
                    tweet = (ex.get("Tweet") or "").strip()
                    question = (ex.get("Question") or "").strip()
                    # Combine tweet and question into single input column
                    input_text = f"Tweet: {tweet}\nQuestion: {question}"
                    answers = ex.get("Answer") or []
                    answer = answers[0].strip() if answers else ""
                    writer.writerow([input_text, answer])
        
        print(f"Saved {len(train_examples)} training examples to {csv_path}")
        exit(0)

    # Use seeded generator for deterministic shuffling
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_train_size, shuffle=True, generator=g)

    lr = args.learning_rate
    all_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in all_params):,} || all params: {sum(p.numel() for p in model.parameters()):,} || trainable%: {100.0*sum(p.numel() for p in all_params)/max(1,sum(p.numel() for p in model.parameters())):.4f}")
    
    optimizer_all = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * total_epochs
    warmup_steps = int(0.05 * total_steps)
    sched_all = get_linear_schedule_with_warmup(optimizer_all, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print("\nOptimization setup:")
    print(f"  LR: {lr} | total_steps: {total_steps} | warmup_steps: {warmup_steps} | steps/epoch: {len(train_loader)}")

    if args.dataset == "squad" and not args.skip_all_eval:
        print("\n" + "="*60)
        print("INITIAL EVALUATION (EM + F1, before training)")
        print("="*60)
        try:
            initial_em, initial_f1 = evaluate_squad_em_f1(
                model, tokenizer, fixed_val_eval,
                max_src_len=args.max_length, max_gen_len=args.max_target_length,
                num_samples=min(200, len(fixed_val_eval)), debug=True,
                dataset_name="SQuAD"
            )
            print(f"Initial EM: {initial_em:.2f}%, F1: {initial_f1:.2f}%")
        except Exception as e:
            print(f"[warn] Initial evaluation skipped: {e}")
    elif args.dataset == "tweetqa" and not args.skip_all_eval:
        print("\n" + "="*60)
        print("INITIAL EVALUATION (EM + F1, before training)")
        print("="*60)
        try:
            initial_em, initial_f1 = evaluate_tweetqa_em_f1(
                model, tokenizer, fixed_val_eval,
                max_src_len=args.max_length, max_gen_len=args.max_target_length,
                num_samples=min(200, len(fixed_val_eval)), debug=True
            )
            print(f"Initial EM: {initial_em:.2f}%, F1: {initial_f1:.2f}%")
        except Exception as e:
            print(f"[warn] Initial TweetQA evaluation skipped: {e}")
    
    # Clear cache after initial evaluation
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Load adversarial variants for SDBN-PL/SDBN-PS mode
    # For single CSV mode, load once before training loop
    # For multi-CSV mode, will load per-epoch inside the loop
    adversarial_variants = None
    if args.perturbation in 'sdbn-p' and not args.multi_csv:
        adversarial_variants = load_adversarial_csv(output_dir, args)
    
    # Validate multi_csv settings
    if args.multi_csv:
        if args.perturbation not in 'sdbn-p':
            raise ValueError("multi_csv flag can only be used with sdbn-p or sdbn-p perturbation")
        csv_start_epoch = args.warmup_epochs + 1
        csv_end_epoch = total_epochs
        print(f"[multi_csv] Enabled: will load epoch-specific CSVs from epoch {csv_start_epoch} to {csv_end_epoch}")

    history = {"train_loss": [], "train_metric": [], "val_metric": [], "train_f1": [], "val_f1": [], "metric_name": ""}
    metric_name = "EM"
    history["metric_name"] = metric_name
    
    val_eval_subset = None
    
    # Best checkpoint tracking (save best F1 model)
    best_val_metric = -float('inf')  # Track best validation metric (F1)
    best_epoch = 0
    best_model_state = None  # Store best model state dict in memory

    for epoch in range(1, total_epochs + 1):
        print("\n" + "="*60)
        print(f"Epoch {epoch}/{total_epochs}")
        
        # For multi-CSV mode: load epoch-specific adversarial variants
        if args.multi_csv and args.perturbation == 'sdbn-p':
            csv_start_epoch = args.warmup_epochs + 1
            csv_end_epoch = total_epochs
            if csv_start_epoch <= epoch <= csv_end_epoch:
                # epoch_num is the adversarial epoch (1-based: 1st adv epoch = 1, 2nd = 2, etc.)
                adv_epoch_num = epoch - args.warmup_epochs - 1
                try:
                    adversarial_variants = load_adversarial_csv(output_dir, args, epoch_num=adv_epoch_num)
                    print(f"[multi_csv] Loaded e{adv_epoch_num}.csv for epoch {epoch}")
                except FileNotFoundError as e:
                    print(f"[multi_csv] Warning: {e}")
                    adversarial_variants = None
            else:
                adversarial_variants = None
        
        if epoch <= args.warmup_epochs:
            use_adversarial = False
            print(f"Training mode: Normal (warmup epoch {epoch}/{args.warmup_epochs})")
        else:
            use_adversarial = args.perturbation is not None
            epoch_in_main = epoch - args.warmup_epochs
            if use_adversarial:
                print(f"Training mode: Adversarial ({args.perturbation}) - epoch {epoch_in_main}/{args.epochs}")
            else:
                print(f"Training mode: Normal - epoch {epoch_in_main}/{args.epochs}")

        opt, sch = optimizer_all, sched_all
            
        
        avg_loss, peak_alloc_gib, peak_reserved_gib = train_one_epoch(
            model, train_loader, opt, sch, device, epoch, max_grad_norm=1.0,
            use_adversarial=use_adversarial,
            perturbation_type=args.perturbation,
            adversarial_variants=adversarial_variants,
            tokenizer=tokenizer,
            max_source_len=args.max_length,
            max_target_len=args.max_target_length,
            dataset=args.dataset,
            mem_peak_scope=args.mem_peak_scope,
        )

        # Keep a simple, stable summary line (GiB).
        if device.type == 'cuda':
            print(f"Peak GPU memory: allocated={peak_alloc_gib:.2f} GiB, reserved={peak_reserved_gib:.2f} GiB")
        
        history["train_loss"].append(avg_loss)

        train_eval_subset = (train_examples.select(range(min(100, len(train_examples)))))
        val_eval_subset   = (fixed_val_eval.select(range(min(200, len(fixed_val_eval)))))
        if epoch == total_epochs:
            final_val_subset = val_eval_subset

        if args.dataset == "squad":
            if args.skip_all_eval:
                train_em, train_f1, val_em, val_f1 = 0.0, 0.0, 0.0, 0.0
                history["train_metric"].append(train_em)
                history["val_metric"].append(val_em)
                history["train_f1"].append(train_f1)
                history["val_f1"].append(val_f1)
                print("[skip] All evaluation skipped for faster training")
            else:
                try:
                    if args.skip_train_eval:
                        train_em, train_f1 = 0.0, 0.0
                    else:
                        train_em, train_f1 = evaluate_squad_em_f1(
                            model, tokenizer, train_eval_subset,
                            max_src_len=args.max_length, max_gen_len=args.max_target_length,
                            num_samples=len(train_eval_subset), debug=False,
                            dataset_name="SQuAD"
                        )
                    val_em, val_f1 = evaluate_squad_em_f1(
                        model, tokenizer, val_eval_subset,
                        max_src_len=args.max_length, max_gen_len=args.max_target_length,
                        num_samples=len(val_eval_subset), debug=False,
                        dataset_name="SQuAD"
                    )
                except Exception as e:
                    print(f"[warn] squad evaluation failed: {e}")
                    train_em, train_f1, val_em, val_f1 = 0.0, 0.0, 0.0, 0.0
                history["train_metric"].append(train_em)
                history["val_metric"].append(val_em)
                history["train_f1"].append(train_f1)
                history["val_f1"].append(val_f1)
                if args.skip_train_eval:
                    print(f"Val EM: {val_em:.2f}%, F1: {val_f1:.2f}% (train eval skipped)")
                else:
                    print(f"Train EM: {train_em:.2f}%, F1: {train_f1:.2f}% | Val EM: {val_em:.2f}%, F1: {val_f1:.2f}%")
                
                # Track best model based on val F1 (save state in memory)
                if val_f1 > best_val_metric:
                    best_val_metric = val_f1
                    best_epoch = epoch
                    # Store model state in memory (will save as final at end)
                    from peft import get_peft_model_state_dict
                    best_model_state = {k: v.cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
                    print(f"    [★] New best F1! (epoch {epoch}, F1={val_f1:.2f}%) - will save this as final model")
        
        elif args.dataset == "tweetqa":
            if args.skip_all_eval:
                train_em, train_f1, val_em, val_f1 = 0.0, 0.0, 0.0, 0.0
                history["train_metric"].append(train_em)
                history["val_metric"].append(val_em)
                history["train_f1"].append(train_f1)
                history["val_f1"].append(val_f1)
                print("[skip] All evaluation skipped for faster training")
            else:
                try:
                    if args.skip_train_eval:
                        train_em, train_f1 = 0.0, 0.0
                    else:
                        train_em, train_f1 = evaluate_tweetqa_em_f1(
                            model, tokenizer, train_eval_subset,
                            max_src_len=args.max_length, max_gen_len=args.max_target_length,
                            num_samples=len(train_eval_subset), debug=False
                        )
                    val_em, val_f1 = evaluate_tweetqa_em_f1(
                        model, tokenizer, val_eval_subset,
                        max_src_len=args.max_length, max_gen_len=args.max_target_length,
                        num_samples=len(val_eval_subset), debug=False
                    )
                except Exception as e:
                    print(f"[warn] TweetQA evaluation failed: {e}")
                    train_em, train_f1, val_em, val_f1 = 0.0, 0.0, 0.0, 0.0
                history["train_metric"].append(train_em)
                history["val_metric"].append(val_em)
                history["train_f1"].append(train_f1)
                history["val_f1"].append(val_f1)
                if args.skip_train_eval:
                    print(f"Val EM: {val_em:.2f}%, F1: {val_f1:.2f}% (train eval skipped)")
                else:
                    print(f"Train EM: {train_em:.2f}%, F1: {train_f1:.2f}% | Val EM: {val_em:.2f}%, F1: {val_f1:.2f}%")
                
                # Track best model based on val F1 (save state in memory)
                if val_f1 > best_val_metric:
                    best_val_metric = val_f1
                    best_epoch = epoch
                    # Store model state in memory (will save as final at end)
                    from peft import get_peft_model_state_dict
                    best_model_state = {k: v.cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
                    print(f"    [★] New best F1! (epoch {epoch}, F1={val_f1:.2f}%) - will save this as final model")
        
        else:
            history["train_metric"].append(None)
            history["val_metric"].append(None)
            history["train_f1"].append(None)
            history["val_f1"].append(None)

        #if (epoch % 2) == 0:
        #    ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}_seed_{args.seed}.pt")
        #    save_lora_gen_model(model, ckpt_path)

    # Add best epoch info to metrics
    if args.dataset in ["squad", "tweetqa"] and best_epoch >= 1:
        history["best_epoch"] = best_epoch
        history["best_val_f1"] = best_val_metric
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"metrics_seed_{args.seed}.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    final_model_path = os.path.join(output_dir, f"model_seed_{args.seed}.pt")
    
    # Save model: best checkpoint (if eval enabled), otherwise final epoch
    if args.dataset in ["squad", "tweetqa"] and best_epoch >= 1 and best_model_state is not None and not args.skip_all_eval:
        # Restore and save best model
        metric_name = "F1"
        print(f"\n{'='*60}")
        print(f"[★] SAVING BEST MODEL (not final epoch)")
        print(f"[★] Best performance: Epoch {best_epoch}/{total_epochs} with {metric_name}={best_val_metric:.2f}")
        print(f"{'='*60}")
        from peft import set_peft_model_state_dict
        best_model_state_gpu = {k: v.to(next(model.parameters()).device) for k, v in best_model_state.items()}
        set_peft_model_state_dict(model, best_model_state_gpu)
        save_lora_gen_model(model, final_model_path)
        print(f"[✓] Saved best model from epoch {best_epoch} (trained {total_epochs} epochs total)")
    else:
        # For other datasets or skip_all_eval: save final epoch model
        save_lora_gen_model(model, final_model_path)
        if args.skip_all_eval:
            print(f"\n[✓] Evaluation skipped - saved final epoch {total_epochs} model")
        else:
            print(f"\n[✓] Saved final epoch {total_epochs} model")
    
    #tokenizer.save_pretrained(output_dir)
    print(f"[✓] Done. Results saved in: {output_dir}")
    
    # === VERIFICATION ===
    print("\n" + "="*60)
    print("VERIFICATION: Loading saved model to prove save/load works")
    print("="*60)
    
    try:
        original_device = next(model.parameters()).device
        last_val_metric = history["val_metric"][-1] if history["val_metric"] else None
        last_val_f1 = history["val_f1"][-1] if history.get("val_f1") else None
        
        # For squad/tweetqa: compare against best epoch, not last epoch
        if args.dataset in ["squad", "tweetqa"] and best_epoch >= 1 and not args.skip_all_eval:
            expected_metric = best_val_metric
            expected_epoch = best_epoch
            metric_name = "F1"
            print(f"Comparing against BEST epoch {best_epoch} (F1={best_val_metric:.2f}%)")
        else:
            expected_metric = last_val_f1 if args.dataset in ["squad", "tweetqa"] else last_val_metric
            expected_epoch = total_epochs
        
        loaded_model = load_lora_gen_model(
            final_model_path,
            model_full_name,
            lora_rank=args.lora_rank,
            device=original_device,
        )
        
        print("Testing loaded model performance on EXACT same samples as last epoch...")
        try:
            test_subset = final_val_subset
            print(f"Using same {len(test_subset)} samples from last epoch validation")
        except NameError:
            print("[warn] Using fallback validation subset")
            test_subset = fixed_val_eval.select(range(min(200, len(fixed_val_eval))))
        
        loaded_metric = None
        verification_status = "UNKNOWN"
        if args.dataset in ["squad", "tweetqa"]:
            try:
                loaded_em, loaded_f1 = evaluate_squad_em_f1(
                    loaded_model, tokenizer, test_subset,
                    max_src_len=args.max_length, max_gen_len=args.max_target_length,
                    num_samples=len(test_subset), debug=False,
                    dataset_name="SQuAD"
                ) if args.dataset != "tweetqa" else evaluate_tweetqa_em_f1(
                    loaded_model, tokenizer, test_subset,
                    max_src_len=args.max_length, max_gen_len=args.max_target_length,
                    num_samples=len(test_subset), debug=False
                )
                print(f"\nPerformance Comparison ({args.dataset}):")
                if best_epoch > 0 and not args.skip_all_eval:
                    print(f"    Best epoch {best_epoch} Val F1: {best_val_metric:.2f}%")
                else:
                    print(f"    Last training Val F1: {last_val_f1:.2f}%" if last_val_f1 else "    Last training Val F1: N/A")
                print(f"    Loaded model EM: {loaded_em:.2f}%, F1: {loaded_f1:.2f}%")
                
                # Compare against best epoch or last epoch
                compare_f1 = best_val_metric if (best_epoch > 0 and not args.skip_all_eval) else (last_val_f1 or 0)
                if compare_f1 and abs(loaded_f1 - compare_f1) > 2.0:
                    print(f"    [⚠] PERFORMANCE MISMATCH! Difference: {abs(loaded_f1 - compare_f1):.2f}%")
                    verification_status = "FAILED - Performance Mismatch"
                elif compare_f1 and abs(loaded_f1 - compare_f1) > 0.5:
                    print(f"    [⚠] Minor performance difference: {abs(loaded_f1 - compare_f1):.2f}%")
                    verification_status = "PARTIAL - Minor Mismatch"
                else:
                    print(f"    [✓] Performance preserved within acceptable range")
                    verification_status = "SUCCESSFUL"
            except Exception as e:
                print(f"[⚠] {args.dataset} evaluation failed: {e}")
                verification_status = "PARTIAL - Load OK, Eval Failed"
        else:
            verification_status = "SUCCESSFUL - Basic Load Test"
            
        original_params = sum(p.numel() for p in model.parameters())
        loaded_params = sum(p.numel() for p in loaded_model.parameters())
        original_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        loaded_trainable = sum(p.numel() for p in loaded_model.parameters() if p.requires_grad)
        
        print(f"\nArchitecture Comparison:")
        print(f"    Total params - Original: {original_params:,}, Loaded: {loaded_params:,}")
        print(f"    Trainable params - Original: {original_trainable:,}, Loaded: {loaded_trainable:,}")
        
        if original_params == loaded_params and original_trainable == loaded_trainable:
            print(f"    [✓] Parameter counts match perfectly!")
        else:
            print(f"    [⚠] Parameter count mismatch - check LoRA rank and encoder freezing logic")
            if "SUCCESSFUL" in verification_status:
                verification_status = "FAILED - Architecture Mismatch"
        
        print(f"\n=== VERIFICATION {verification_status.upper()} ===")
        if "FAILED" in verification_status:
            print(f"The save/load process has issues that need to be fixed.")
        elif "SUCCESSFUL" in verification_status:
            print(f"Save/load process is working correctly.")
            
    except Exception as e:
        print(f"\n[✗] VERIFICATION FAILED: {e}")
        print(f"    Model was saved but could not be loaded properly.")
        verification_status = "FAILED - Load Error"
    
    print("\n" + "="*60)
    print(f"Training and verification complete. All files in: {output_dir}")
    if "FAILED" in verification_status:
        print("⚠ IMPORTANT: Save/load verification failed. Model may not restore correctly.")
    print("="*60)
    
    