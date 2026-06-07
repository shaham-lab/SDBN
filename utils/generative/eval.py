"""
noise_gen.py
Evaluate SQuAD models with LoRA on noisy data
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, set_peft_model_state_dict
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
import utils
import utils_gen
import pandas as pd
import re
import argparse
from os.path import join, exists
from os import environ
import json
from collections import Counter
import string
from datasets import load_dataset, Dataset as HFDataset

# Try to import rouge_score for SAMSum evaluation
try:
    from rouge_score import rouge_scorer
    rouge_available = True
except ImportError:
    rouge_available = False
    print("[Warning] rouge_score not available. Install with: pip install rouge-score")


# Model configurations
MODELS = ['deberta']
ALGS = ['lora']  # Only LoRA for now, can add others later
PERTURBATIONS = ['sdbn-p']  # Training perturbations
TRAIN_SIZES = [1000, 2000]  # Different training set sizes

models_map = {
    'llama': 'meta-llama/Llama-3.2-1B',
    'llama-7b': 'meta-llama/Llama-2-7b-hf',
    'qwen': 'Qwen/Qwen2.5-7B',
}

def is_decoder_only_model(model_name: str) -> bool:
    """Check if model is decoder-only (LLaMA/Qwen) vs extractive (DeBERTa)"""
    name = model_name.lower()
    return 'llama' in name or 'qwen' in name

SEEDS = 10  # Number of seeds to evaluate
noise_levels = 5
batch_size = 1000
rank = 4

# Noise types for evaluation
NOISE_TYPES = ['replace_word', 'delete_word', 'swap_word', 'homophone', 'sms', 
               'cyrillic', 'case', 'pronoun_swap', 'add_space', 'remove_space',
               'delete_char', 'swap_char', 'double_char', 'keyboard_char']


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
                item[key] = val[idx]
            else:
                item[key] = torch.tensor(val[idx])
        return item


def prepare_validation_features(examples, tokenizer, max_length=384, doc_stride=128):
    """Prepare validation features for DeBERTa"""
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


def noise_squad_data(examples, noise_level=1, noise_type=None):
    """Apply noise to SQuAD dataset and return as HuggingFace Dataset"""
    if noise_level < 1:
        print("Invalid noise_level:", noise_level)
        sys.exit(1)
    
    noisy_questions = []
    noisy_contexts = []
    ids = []
    answers = []
    
    for i in tqdm(range(len(examples)), desc=f"Applying {noise_type} noise"):
        question = examples[i]['question']
        context = examples[i]['context']
        
        # Apply noise to both question and context
        for _ in range(noise_level):
            if noise_type == 'replace_word':
                question = utils.replace_word_embedding(question)
                context = utils.replace_word_embedding(context)
            elif noise_type == 'delete_word':
                question = utils.delete_random_word(question)
                context = utils.delete_random_word(context)
            elif noise_type == 'swap_word':
                question = utils.swap_two_random_words(question)
                context = utils.swap_two_random_words(context)
            elif noise_type == 'homophone':
                question = utils.noise_homophones(question, p=0.2)
                context = utils.noise_homophones(context, p=0.2)
            elif noise_type == 'sms':
                question = utils.sms_chat_contraction_noise(question)
                context = utils.sms_chat_contraction_noise(context)
            elif noise_type == 'cyrillic':
                question = utils.cyrillic_homoglyph_noise(question)
                context = utils.cyrillic_homoglyph_noise(context)
            elif noise_type == 'case':
                question = utils.remove_cues_noise(question)
                context = utils.remove_cues_noise(context)
            elif noise_type == 'pronoun_swap':
                question = utils.pronoun_swap_noise(question)
                context = utils.pronoun_swap_noise(context)
            elif noise_type == 'add_space':
                question = utils.add_random_space(question)
                context = utils.add_random_space(context)
            elif noise_type == 'remove_space':
                question = utils.remove_random_space(question)
                context = utils.remove_random_space(context)
            elif noise_type == 'delete_char':
                question = utils.delete_char(question)
                context = utils.delete_char(context)
            elif noise_type == 'swap_char':
                question = utils.swap_adj_chars(question)
                context = utils.swap_adj_chars(context)
            elif noise_type == 'double_char':
                question = utils.double_char(question)
                context = utils.double_char(context)
            elif noise_type == 'keyboard_char':
                question = utils.keyboard_char(question)
                context = utils.keyboard_char(context)
            else:
                raise ValueError(f"Invalid noise type: {noise_type}")
        
        noisy_questions.append(question)
        noisy_contexts.append(context)
        ids.append(examples[i]['id'])
        answers.append(examples[i]['answers'])
    
    # Create HuggingFace Dataset from noisy data
    noisy_data = {
        'id': ids,
        'question': noisy_questions,
        'context': noisy_contexts,
        'answers': answers
    }
    
    return HFDataset.from_dict(noisy_data)


def noise_samsum_data(examples, noise_level=1, noise_type=None):
    """Apply noise to SAMSum dataset and return as HuggingFace Dataset"""
    if noise_level < 1:
        print("Invalid noise_level:", noise_level)
        sys.exit(1)
    
    noisy_dialogues = []
    summaries = []
    ids = []
    
    for i in tqdm(range(len(examples)), desc=f"Applying {noise_type} noise to SAMSum"):
        dialogue = examples[i]['dialogue']
        summary = examples[i]['summary']
        example_id = examples[i].get('id', str(i))
        
        # Apply noise only to dialogue (not summary - that's the target)
        for _ in range(noise_level):
            if noise_type == 'replace_word':
                dialogue = utils.replace_word_embedding(dialogue)
            elif noise_type == 'delete_word':
                dialogue = utils.delete_random_word(dialogue)
            elif noise_type == 'swap_word':
                dialogue = utils.swap_two_random_words(dialogue)
            elif noise_type == 'homophone':
                dialogue = utils.noise_homophones(dialogue, p=0.2)
            elif noise_type == 'sms':
                dialogue = utils.sms_chat_contraction_noise(dialogue)
            elif noise_type == 'cyrillic':
                dialogue = utils.cyrillic_homoglyph_noise(dialogue)
            elif noise_type == 'case':
                dialogue = utils.remove_cues_noise(dialogue)
            elif noise_type == 'pronoun_swap':
                dialogue = utils.pronoun_swap_noise(dialogue)
            elif noise_type == 'add_space':
                dialogue = utils.add_random_space(dialogue)
            elif noise_type == 'remove_space':
                dialogue = utils.remove_random_space(dialogue)
            elif noise_type == 'delete_char':
                dialogue = utils.delete_char(dialogue)
            elif noise_type == 'swap_char':
                dialogue = utils.swap_adj_chars(dialogue)
            elif noise_type == 'double_char':
                dialogue = utils.double_char(dialogue)
            elif noise_type == 'keyboard_char':
                dialogue = utils.keyboard_char(dialogue)
            else:
                raise ValueError(f"Invalid noise type: {noise_type}")
        
        noisy_dialogues.append(dialogue)
        summaries.append(summary)
        ids.append(example_id)
    
    # Create HuggingFace Dataset from noisy data
    noisy_data = {
        'id': ids,
        'dialogue': noisy_dialogues,
        'summary': summaries
    }
    
    return HFDataset.from_dict(noisy_data)


def noise_tweetqa_data(examples, noise_level=1, noise_type=None):
    """Apply noise to TweetQA dataset and return as HuggingFace Dataset"""
    if noise_level < 1:
        print("Invalid noise_level:", noise_level)
        sys.exit(1)
    
    noisy_tweets = []
    noisy_questions = []
    answers = []
    ids = []
    
    for i in tqdm(range(len(examples)), desc=f"Applying {noise_type} noise to TweetQA"):
        tweet = examples[i]['Tweet']
        question = examples[i]['Question']
        answer = examples[i]['Answer']
        example_id = str(i)
        
        # Apply noise to both tweet and question
        for _ in range(noise_level):
            if noise_type == 'replace_word':
                tweet = utils.replace_word_embedding(tweet)
                question = utils.replace_word_embedding(question)
            elif noise_type == 'delete_word':
                tweet = utils.delete_random_word(tweet)
                question = utils.delete_random_word(question)
            elif noise_type == 'swap_word':
                tweet = utils.swap_two_random_words(tweet)
                question = utils.swap_two_random_words(question)
            elif noise_type == 'homophone':
                tweet = utils.noise_homophones(tweet, p=0.2)
                question = utils.noise_homophones(question, p=0.2)
            elif noise_type == 'sms':
                tweet = utils.sms_chat_contraction_noise(tweet)
                question = utils.sms_chat_contraction_noise(question)
            elif noise_type == 'cyrillic':
                tweet = utils.cyrillic_homoglyph_noise(tweet)
                question = utils.cyrillic_homoglyph_noise(question)
            elif noise_type == 'case':
                tweet = utils.remove_cues_noise(tweet)
                question = utils.remove_cues_noise(question)
            elif noise_type == 'pronoun_swap':
                tweet = utils.pronoun_swap_noise(tweet)
                question = utils.pronoun_swap_noise(question)
            elif noise_type == 'add_space':
                tweet = utils.add_random_space(tweet)
                question = utils.add_random_space(question)
            elif noise_type == 'remove_space':
                tweet = utils.remove_random_space(tweet)
                question = utils.remove_random_space(question)
            elif noise_type == 'delete_char':
                tweet = utils.delete_char(tweet)
                question = utils.delete_char(question)
            elif noise_type == 'swap_char':
                tweet = utils.swap_adj_chars(tweet)
                question = utils.swap_adj_chars(question)
            elif noise_type == 'double_char':
                tweet = utils.double_char(tweet)
                question = utils.double_char(question)
            elif noise_type == 'keyboard_char':
                tweet = utils.keyboard_char(tweet)
                question = utils.keyboard_char(question)
            else:
                raise ValueError(f"Invalid noise type: {noise_type}")
        
        noisy_tweets.append(tweet)
        noisy_questions.append(question)
        answers.append(answer)
        ids.append(example_id)
    
    # Create HuggingFace Dataset from noisy data
    noisy_data = {
        'Tweet': noisy_tweets,
        'Question': noisy_questions,
        'Answer': answers,
        'id': ids
    }
    
    return HFDataset.from_dict(noisy_data)


def postprocess_qa_predictions(start_logits, end_logits, features, examples, tokenizer, n_best=20, max_answer_length=30):
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


def evaluate_model(model, eval_dataloader, eval_dataset_for_model, eval_examples, tokenizer, device):
    """Evaluate model on dataset"""
    model.eval()
    all_start_logits = []
    all_end_logits = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'example_id'}
            outputs = model(**inputs)
            
            all_start_logits.extend(outputs.start_logits.cpu().numpy())
            all_end_logits.extend(outputs.end_logits.cpu().numpy())
    
    all_start_logits = np.array(all_start_logits)
    all_end_logits = np.array(all_end_logits)
    
    # Convert dataset to list of features for postprocessing efficiently
    # Convert to dict format first to avoid slow indexing
    dataset_dict = eval_dataset_for_model.to_dict() if hasattr(eval_dataset_for_model, 'to_dict') else eval_dataset_for_model
    
    features = []
    num_examples = len(dataset_dict['input_ids'])
    for i in range(num_examples):
        feature = {
            'input_ids': dataset_dict['input_ids'][i],
            'attention_mask': dataset_dict['attention_mask'][i],
            'example_id': dataset_dict['example_id'][i],
            'offset_mapping': dataset_dict['offset_mapping'][i]
        }
        features.append(feature)
    
    predictions = postprocess_qa_predictions(
        all_start_logits, 
        all_end_logits, 
        features, 
        eval_examples,
        tokenizer
    )
    
    references = {}
    for example in eval_examples:
        if len(example['answers']['text']) > 0:
            references[example['id']] = example['answers']['text'][0]
        else:
            references[example['id']] = ""
    
    metrics = compute_metrics(predictions, references)
    return metrics


# ============================================================
# GENERATIVE EVALUATION FOR LLaMA (decoder-only models)
# ============================================================

def normalize_answer_generative(s: str) -> str:
    """Normalize answer text for SQuAD evaluation (generative)"""
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


def compute_squad_em_generative(prediction: str, ground_truth: str) -> int:
    """Compute exact match for SQuAD (generative)"""
    return int(normalize_answer_generative(prediction) == normalize_answer_generative(ground_truth))


def compute_squad_f1_generative(prediction: str, ground_truth: str) -> float:
    """Compute F1 score for SQuAD (generative)"""
    pred_tokens = normalize_answer_generative(prediction).split()
    truth_tokens = normalize_answer_generative(ground_truth).split()
    
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


def load_lora_llama_model(model_path, model_name, lora_rank=4, device='cuda:0'):
    """Load LoRA model for LLaMA/Qwen (decoder-only)"""
    print(f"Loading decoder-only model from: {model_path}")
    
    state_dict = torch.load(model_path, map_location='cpu')
    
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    _mn = model_name.lower()
    _dtype = torch.bfloat16 if ("7b" in _mn or "8b" in _mn or "qwen" in _mn) else None
    base_model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype=_dtype)
    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        modules_to_save=[],
    )
    model = get_peft_model(base_model, lora_config)
    
    # Load LoRA weights
    if "lora" in state_dict:
        set_peft_model_state_dict(model, state_dict["lora"])
        print(f"[✓] Loaded LoRA weights: {len(state_dict['lora'])} tensors")
    else:
        print("[⚠] No LoRA state found in saved model")
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


@torch.no_grad()
def evaluate_squad_generative(model, tokenizer, examples, device, max_src_len=384, max_gen_len=32, 
                              batch_size=16, debug=False):
    """
    Evaluate SQuAD using EM and F1 metrics for decoder-only models (LLaMA).
    Format: Context: <context>\nQuestion: <question>\nAnswer: 
    
    Uses BATCHED generation for speed (with left-padding for decoder-only models).
    """
    model.eval()
    
    # Prepare all prompts and ground truths
    prompts = []
    ground_truths = []
    valid_indices = []
    
    for i, ex in enumerate(examples):
        context = (ex.get("context") or "").strip()
        question = (ex.get("question") or "").strip()
        answers = ex.get("answers") or {}
        
        if not context or not question:
            continue
        
        answer_texts = answers.get("text", [])
        if not answer_texts:
            continue
        
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        prompts.append(prompt)
        ground_truths.append(answer_texts[0].strip())
        valid_indices.append(i)
    
    if len(prompts) == 0:
        return {'exact_match': 0.0, 'f1': 0.0}
    
    # Set up left-padding for batched decoder-only generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    em_scores = []
    f1_scores = []
    
    # Process in batches
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Evaluating SQuAD (batched)"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        batch_truths = ground_truths[start_idx:end_idx]
        
        # Tokenize batch with left-padding
        inputs = tokenizer(
            batch_prompts,
            max_length=max_src_len,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate for entire batch
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Process each output in batch
        for j, (out, truth) in enumerate(zip(out_ids, batch_truths)):
            # Get prompt length for this example (accounting for left-padding)
            prompt_len = inputs["input_ids"][j].shape[0]
            
            # Slice off prompt
            pred_ids = out[prompt_len:]
            prediction = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            
            # Stop at newline or common continuation markers
            for stop_marker in ['\n', 'Context:', 'Question:', 'Explanation:', 'Note:']:
                if stop_marker in prediction:
                    prediction = prediction.split(stop_marker)[0].strip()
                    break
            
            # Compute metrics
            em = compute_squad_em_generative(prediction, truth)
            f1 = compute_squad_f1_generative(prediction, truth)
            
            em_scores.append(em)
            f1_scores.append(f1)
            
            if debug and (start_idx + j) < 5:
                print(f"\nSQuAD example {start_idx + j + 1}:")
                print(f"  Ground truth: {truth}")
                print(f"  Prediction: {prediction}")
                print(f"  EM: {em}, F1: {f1:.2f}")
                print("-" * 60)
    
    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    
    em_score = 100.0 * sum(em_scores) / len(em_scores)
    f1_score = 100.0 * sum(f1_scores) / len(f1_scores)
    
    print(f"\nSQuAD Results: EM={em_score:.2f}%, F1={f1_score:.2f}% ({len(em_scores)} examples)")
    
    return {'exact_match': float(em_score), 'f1': float(f1_score)}


# ============================================================
# TWEETQA EVALUATION (EM and F1 scores)
# ============================================================

@torch.no_grad()
def evaluate_tweetqa_generative(model, tokenizer, examples, device, max_src_len=512, max_gen_len=32,
                                batch_size=16, debug=False):
    """
    Evaluate TweetQA using EM and F1 metrics for decoder-only models (LLaMA).
    TweetQA structure: Tweet, Question, Answer (list)
    Format: Tweet: <tweet>\nQuestion: <question>\nAnswer:
    
    Uses BATCHED generation for speed (with left-padding for decoder-only models).
    """
    model.eval()
    
    # Prepare all prompts and ground truths
    prompts = []
    ground_truths = []
    valid_indices = []
    
    for i, ex in enumerate(examples):
        tweet = (ex.get("Tweet") or "").strip()
        question = (ex.get("Question") or "").strip()
        answers = ex.get("Answer") or []
        
        if not tweet or not question:
            continue
        
        # TweetQA Answer is a list - get first answer, or handle single answer
        if not answers:
            continue
        answer_texts = answers if isinstance(answers, list) else [answers]
        answer_texts = [a.strip() for a in answer_texts if a]
        if not answer_texts:
            continue
        
        prompt = f"Tweet: {tweet}\nQuestion: {question}\nAnswer:"
        prompts.append(prompt)
        ground_truths.append(answer_texts)  # Keep list of all valid answers
        valid_indices.append(i)
    
    if len(prompts) == 0:
        return {'exact_match': 0.0, 'f1': 0.0}
    
    # Set up left-padding for batched decoder-only generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    em_scores = []
    f1_scores = []
    
    # Process in batches
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Evaluating TweetQA (batched)"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        batch_truths = ground_truths[start_idx:end_idx]
        
        # Tokenize batch with left-padding
        inputs = tokenizer(
            batch_prompts,
            max_length=max_src_len,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate for entire batch
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Process each output in batch
        for j, (out, truth_list) in enumerate(zip(out_ids, batch_truths)):
            # Get prompt length for this example (accounting for left-padding)
            prompt_len = inputs["input_ids"][j].shape[0]
            
            # Slice off prompt
            pred_ids = out[prompt_len:]
            prediction = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            
            # Stop at newline or common continuation markers
            for stop_marker in ['\n', 'Tweet:', 'Question:', 'Explanation:', 'Note:']:
                if stop_marker in prediction:
                    prediction = prediction.split(stop_marker)[0].strip()
                    break
            
            # Compute metrics: max over all reference answers (standard SQuAD protocol)
            em = max(compute_squad_em_generative(prediction, truth) for truth in truth_list) if truth_list else 0
            f1 = max(compute_squad_f1_generative(prediction, truth) for truth in truth_list) if truth_list else 0.0
            
            em_scores.append(em)
            f1_scores.append(f1)
            
            if debug and (start_idx + j) < 5:
                print(f"\nTweetQA example {start_idx + j + 1}:")
                print(f"  Ground truth: {truth_list}")
                print(f"  Prediction: {prediction}")
                print(f"  EM: {em}, F1: {f1:.2f}")
                print("-" * 60)
    
    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    
    em_score = 100.0 * sum(em_scores) / len(em_scores)
    f1_score = 100.0 * sum(f1_scores) / len(f1_scores)
    
    print(f"\nTweetQA Results: EM={em_score:.2f}%, F1={f1_score:.2f}% ({len(em_scores)} examples)")
    
    return {'exact_match': float(em_score), 'f1': float(f1_score)}


# ============================================================
# SAMSUM EVALUATION (ROUGE scores)
# ============================================================

@torch.no_grad()
def evaluate_samsum_rouge(model, tokenizer, examples, device, max_src_len=512, max_gen_len=128,
                          batch_size=8, debug=False):
    """
    Evaluate SAMSum using ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    Uses batched generation for faster evaluation (decoder-only models with left-padding).
    Returns: dict with 'rouge1', 'rouge2', 'rougeL' as F1 percentages
    """
    if not rouge_available:
        raise ImportError("rouge_score not available. Install with: pip install rouge-score")
    
    model.eval()
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Collect all valid examples
    all_dialogues = []
    all_summaries = []
    for ex in examples:
        dialogue = (ex.get("dialogue") or "").strip()
        summary = (ex.get("summary") or "").strip()
        if dialogue and summary:
            all_dialogues.append(dialogue)
            all_summaries.append(summary)
    
    if not all_dialogues:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    # Set up left-padding for batched decoder-only generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    # Process in batches
    num_batches = (len(all_dialogues) + batch_size - 1) // batch_size
    all_predictions = []
    
    for batch_idx in tqdm(range(num_batches), desc="Evaluating ROUGE (SAMSum)"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_dialogues))
        
        batch_dialogues = all_dialogues[start_idx:end_idx]
        
        # Decoder-only: use prompt format
        prompts = [f"Dialogue:\n{d}\n\nSummary:\n" for d in batch_dialogues]
        
        # Tokenize batch with left padding
        inputs = tokenizer(
            prompts,
            max_length=max_src_len,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate for entire batch
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Decode predictions (slice off prompts)
        for i in range(len(batch_dialogues)):
            # The generation starts after the input
            pred_ids = out_ids[i][inputs["input_ids"].shape[1]:]
            prediction = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            
            # Stop at newline or dialogue markers
            for stop_marker in ['\n\n', 'Dialogue:', 'Summary:']:
                if stop_marker in prediction:
                    prediction = prediction.split(stop_marker)[0].strip()
                    break
            
            all_predictions.append(prediction)
    
    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    
    # Compute ROUGE scores
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for i, (summary, prediction) in enumerate(zip(all_summaries, all_predictions)):
        scores = scorer.score(summary, prediction)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
        
        if debug and i < 3:
            print(f"\nSAMSum ex {i+1}")
            print(f"  Dialogue: {all_dialogues[i][:100]}...")
            print(f"  Reference: {summary}")
            print(f"  Prediction: {prediction}")
            print(f"  ROUGE-1: {scores['rouge1'].fmeasure:.4f}, ROUGE-2: {scores['rouge2'].fmeasure:.4f}, ROUGE-L: {scores['rougeL'].fmeasure:.4f}")
            print("-" * 60)
    
    if not rouge1_scores:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    # Average and convert to percentage
    rouge1 = 100.0 * sum(rouge1_scores) / len(rouge1_scores)
    rouge2 = 100.0 * sum(rouge2_scores) / len(rouge2_scores)
    rougeL = 100.0 * sum(rougeL_scores) / len(rougeL_scores)
    
    print(f"\nSAMSum ROUGE Results: R-1={rouge1:.2f}%, R-2={rouge2:.2f}%, R-L={rougeL:.2f}% ({len(rouge1_scores)} examples)")
    
    return {'rouge1': float(rouge1), 'rouge2': float(rouge2), 'rougeL': float(rougeL)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_type', '-nt', type=str, default='replace_word', 
                        choices=NOISE_TYPES, help='Set noise type')
    parser.add_argument('--device', type=str, default='0', help='The device to run the program')
    parser.add_argument('--noise_levels', type=int, default=1, 
                        help='The number of noise levels to test')
    parser.add_argument('--dataset', type=str, default='squad', choices=['squad', 'tweetqa'],
                        help='The dataset to use (squad or tweetqa)')
    parser.add_argument('--model', type=str, default='llama', choices=list(models_map.keys()), help='The model to use')
    parser.add_argument('--train_size', type=int, default=1000, help='Training set size to evaluate')
    parser.add_argument('--public_server', '-ps', default=False, action='store_true', 
                        help='Use public server directory')
    parser.add_argument('--perturbation', '-pt', type=str, default='sdbn-p', choices=['sdbn-p'],
                        help='Perturbation type used during training')
    parser.add_argument('--epsilon', '-eps', type=float, default=None,
                        help='Epsilon value for SDBN')
    parser.add_argument('--lora_rank', type=int, default=4, help='LoRA rank')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds to evaluate')
    parser.add_argument('--testset_size', type=int, default=1000, help='Test set size')
    parser.add_argument('--single_level', action='store_true', 
                        help='Use only the specified noise_levels value instead of looping from 1')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='Evaluation batch size (used for LLaMA generation batches and extractive DataLoader).')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of evaluation examples to score (matches generative.py default=200).')
    parser.add_argument('--max_gen_len', type=int, default=32,
                        help='Max new tokens to generate for LLaMA SQuAD (match generative.py: max_target_length).')
    parser.add_argument('--max_samsum_gen_len', type=int, default=128,
                        help='Max new tokens to generate for SAMSum summaries.')
    parser.add_argument('--root', default='', type=str, help='Root directory for model storage (default: current directory)')
    parser.add_argument('--peft_method', type=str, default='lora', choices=['lora', 'none'], help='PEFT method used during training')
    args = parser.parse_args()

    # Use CLI batch size everywhere below
    batch_size = int(args.batch_size)
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # Set root directory
    if args.root:
        root = args.root
    elif args.public_server:
        root = '.'
    else:
        root = '.'
    
    # Get model name
    model_full_name = models_map[args.model]
    is_generative = is_decoder_only_model(model_full_name)
    
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_full_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ==============================================
    # Dataset-specific loading
    # ==============================================
    if args.dataset == 'squad':
        # Load clean SQuAD validation data
        dataset = load_dataset("squad")
        eval_k = min(args.testset_size, len(dataset["validation"]))
        if eval_k <= 0:
            raise ValueError("testset_size must be >= 1")
        eval_examples_clean = dataset["validation"].select(range(eval_k))
        print(f"[SQuAD] Scoring on first {eval_k} validation examples (total={len(dataset['validation'])})")
    
    elif args.dataset == 'adversarial_qa':
        # Load clean AdversarialQA validation data (same structure as SQuAD)
        dataset = load_dataset("UCLNLP/adversarial_qa", "adversarialQA")
        eval_k = min(args.testset_size, len(dataset["validation"]))
        if eval_k <= 0:
            raise ValueError("testset_size must be >= 1")
        eval_examples_clean = dataset["validation"].select(range(eval_k))
        print(f"[AdversarialQA] Scoring on first {eval_k} validation examples (total={len(dataset['validation'])})")
        
        # Prepare clean dataset (only for extractive models)
        if not is_generative:
            dataset_label = "AdversarialQA" if args.dataset == "adversarial_qa" else "SQuAD"
            print(f"Preparing clean validation dataset (extractive) - {dataset_label}...")
            clean_dataset = eval_examples_clean.map(
                lambda x: prepare_validation_features(x, tokenizer),
                batched=True,
                remove_columns=eval_examples_clean.column_names
            )
            
            clean_dataset_final = SQuADDataset({
                'input_ids': clean_dataset['input_ids'],
                'attention_mask': clean_dataset['attention_mask'],
                'example_id': clean_dataset['example_id']
            })
            
            # Store the full clean dataset for postprocessing
            clean_dataset_with_offsets = clean_dataset
        else:
            dataset_label = "AdversarialQA" if args.dataset == "adversarial_qa" else "SQuAD"
            print(f"Using generative evaluation for LLaMA ({dataset_label})...")
            clean_dataset_final = None
            clean_dataset_with_offsets = None
    
    elif args.dataset == 'subjqa':
        # Load clean SubjQA validation data (books config, same structure as SQuAD)
        dataset = load_dataset("MohammedKamal23/subjqa-complete", "books")
        eval_k = min(args.testset_size, len(dataset["validation"]))
        if eval_k <= 0:
            raise ValueError("testset_size must be >= 1")
        eval_examples_clean = dataset["validation"].select(range(eval_k))
        print(f"[SubjQA] Scoring on first {eval_k} validation examples (total={len(dataset['validation'])})")
        
        # Prepare clean dataset (only for extractive models)
        if not is_generative:
            dataset_label = "SubjQA" if args.dataset == "subjqa" else "SQuAD"
            print(f"Preparing clean validation dataset (extractive) - {dataset_label}...")
            clean_dataset = eval_examples_clean.map(
                lambda x: prepare_validation_features(x, tokenizer),
                batched=True,
                remove_columns=eval_examples_clean.column_names
            )
            
            clean_dataset_final = SQuADDataset({
                'input_ids': clean_dataset['input_ids'],
                'attention_mask': clean_dataset['attention_mask'],
                'example_id': clean_dataset['example_id']
            })
            
            # Store the full clean dataset for postprocessing
            clean_dataset_with_offsets = clean_dataset
        else:
            dataset_label = "SubjQA" if args.dataset == "subjqa" else "SQuAD"
            print(f"Using generative evaluation for LLaMA ({dataset_label})...")
            clean_dataset_final = None
            clean_dataset_with_offsets = None
    
    elif args.dataset == 'tweetqa':
        # Load TweetQA validation data (from parquet files)
        data_files = {
            "validation": "https://huggingface.co/datasets/ucsbnlp/tweet_qa/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
        }
        dataset = load_dataset("parquet", data_files=data_files)
        eval_k = min(args.testset_size, len(dataset["validation"]))
        if eval_k <= 0:
            raise ValueError("testset_size must be >= 1")
        eval_examples_clean = dataset["validation"].select(range(eval_k))
        print(f"[TweetQA] Scoring on first {eval_k} validation examples (total={len(dataset['validation'])})")
        
        # TweetQA only supports generative models (LLaMA)
        if not is_generative:
            raise ValueError("TweetQA only supports generative models (use --model llama)")
        
        print("Using generative evaluation for LLaMA (TweetQA)...")
        clean_dataset_final = None
        clean_dataset_with_offsets = None
    
    elif args.dataset == 'samsum':
        # Load SAMSum test data
        if not rouge_available:
            raise ImportError("rouge_score not available for SAMSum. Install with: pip install rouge-score")
        
        dataset = load_dataset("knkarthick/samsum")
        test_data = dataset["test"]
        eval_k = min(args.testset_size, len(test_data))
        if eval_k <= 0:
            raise ValueError("testset_size must be >= 1")
        eval_examples_clean = [test_data[i] for i in range(eval_k)]
        print(f"[SAMSum] Scoring on first {eval_k} test examples (total={len(test_data)})")
        
        # SAMSum only supports generative models (LLaMA)
        if not is_generative:
            raise ValueError("SAMSum only supports generative models (use --model llama)")
        
        print("Using generative evaluation for LLaMA (SAMSum)...")
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # ==============================================
    # Evaluate across noise levels
    # ==============================================
    noise_range = [args.noise_levels] if args.single_level else range(1, args.noise_levels + 1)
    for current_noise_level in noise_range:
        print(f"\n{'='*50}")
        print(f"NOISE LEVEL: {current_noise_level}")
        print(f"{'='*50}")
        
        # Create noisy dataset based on dataset type
        if args.dataset in ['squad', 'adversarial_qa', 'subjqa', 'tweetqa']:
            if args.dataset == 'tweetqa':
                noisy_examples = noise_tweetqa_data(eval_examples_clean, current_noise_level, args.noise_type)
            else:
                noisy_examples = noise_squad_data(eval_examples_clean, current_noise_level, args.noise_type)
            
            # Prepare noisy dataset (only for extractive models)
            if not is_generative:
                noisy_dataset = noisy_examples.map(
                    lambda x: prepare_validation_features(x, tokenizer),
                    batched=True,
                    remove_columns=noisy_examples.column_names
                )
                noisy_dataset_final = SQuADDataset({
                    'input_ids': noisy_dataset['input_ids'],
                    'attention_mask': noisy_dataset['attention_mask'],
                    'example_id': noisy_dataset['example_id']
                })
                noisy_dataset_with_offsets = noisy_dataset
            else:
                noisy_dataset_final = None
                noisy_dataset_with_offsets = None
        

        
        elif args.dataset == 'samsum':
            # Convert list to HF Dataset for noise function
            clean_hf = HFDataset.from_dict({
                'id': [str(i) for i in range(len(eval_examples_clean))],
                'dialogue': [ex['dialogue'] for ex in eval_examples_clean],
                'summary': [ex['summary'] for ex in eval_examples_clean]
            })
            noisy_examples = noise_samsum_data(clean_hf, current_noise_level, args.noise_type)
            # Convert back to list format for evaluate_samsum_rouge
            noisy_examples_list = [noisy_examples[i] for i in range(len(noisy_examples))]
        
        results_map = {}
        
        # Use specified perturbation or all
        perturbations_to_eval = [args.perturbation] if args.perturbation else PERTURBATIONS
        
        # Evaluate each perturbation type
        for perturbation in perturbations_to_eval:
            print(f"\n{'-'*30}")
            print(f"Evaluating {perturbation} models")
            print(f"{'-'*30}")
            
            # Build output directory path - match generative.py structure
            output_dir = join(root, 'results', 'SDBN-P', args.dataset, args.model,
                            f"trainset_size_{args.train_size}", args.peft_method, f"rank_{args.lora_rank}")
            
            if not exists(output_dir):
                print(f"Directory not found: {output_dir}")
                continue
            
            # Initialize metric lists based on dataset
            if args.dataset in ['squad', 'adversarial_qa', 'subjqa', 'tweetqa']:
                clean_em_list = []
                clean_f1_list = []
                noisy_em_list = []
                noisy_f1_list = []
            elif args.dataset == 'samsum':
                clean_r1_list = []
                clean_r2_list = []
                clean_rL_list = []
                noisy_r1_list = []
                noisy_r2_list = []
                noisy_rL_list = []
            
            # Find model files for specified seeds
            model_files = []
            for seed in range(args.seeds):
                # Try different naming conventions
                for pattern in [f"model_seed_{seed}.pt", f"model_{seed}.pt"]:
                    model_path = os.path.join(output_dir, pattern)
                    if exists(model_path):
                        model_files.append((model_path, seed))
                        break
            
            if len(model_files) == 0:
                print(f"No models found in: {output_dir}")
                continue
            
            print(f"Found {len(model_files)} models")
            
            # Evaluate each seed
            for model_path, seed in model_files:
                print(f"\nEvaluating seed {seed}")
                utils.set_random_seed(seed)
                
                # Load model based on type
                try:
                    if is_generative:
                        model, tokenizer = load_lora_llama_model(
                            model_path=model_path,
                            model_name=model_full_name,
                            lora_rank=args.lora_rank,
                            device=device
                        )
                    else:
                        model = utils_gen.load_lora_gen_model(
                            model_name=model_full_name,
                            save_path=model_path,
                            lora_rank=args.lora_rank,
                            device=device
                        )
                except Exception as e:
                    print(f"Error loading model: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Evaluate based on dataset and model type
                if args.dataset in ['squad', 'adversarial_qa', 'subjqa']:
                    if is_generative:
                        # Generative evaluation for LLaMA (batched for speed)
                        clean_metrics = evaluate_squad_generative(
                            model, tokenizer, eval_examples_clean, device,
                            max_src_len=384, max_gen_len=args.max_gen_len, batch_size=batch_size, debug=False
                        )
                        noisy_metrics = evaluate_squad_generative(
                            model, tokenizer, noisy_examples, device,
                            max_src_len=384, max_gen_len=args.max_gen_len, batch_size=batch_size, debug=False
                        )
                    else:
                        # Extractive evaluation for DeBERTa
                        clean_loader = DataLoader(clean_dataset_final, batch_size=batch_size, shuffle=False)
                        noisy_loader = DataLoader(noisy_dataset_final, batch_size=batch_size, shuffle=False)
                        
                        clean_metrics = evaluate_model(model, clean_loader, clean_dataset_with_offsets, 
                                                     eval_examples_clean, tokenizer, device)
                        noisy_metrics = evaluate_model(model, noisy_loader, noisy_dataset_with_offsets, 
                                                     noisy_examples, tokenizer, device)
                    
                    clean_em_list.append(clean_metrics['exact_match'])
                    clean_f1_list.append(clean_metrics['f1'])
                    print(f"Clean - EM: {clean_metrics['exact_match']:.2f}%, F1: {clean_metrics['f1']:.2f}%")
                    
                    noisy_em_list.append(noisy_metrics['exact_match'])
                    noisy_f1_list.append(noisy_metrics['f1'])
                    print(f"Noisy - EM: {noisy_metrics['exact_match']:.2f}%, F1: {noisy_metrics['f1']:.2f}%")
                
                elif args.dataset == 'tweetqa':
                    if is_generative:
                        # Generative evaluation for LLaMA (batched for speed, TweetQA-specific)
                        clean_metrics = evaluate_tweetqa_generative(
                            model, tokenizer, eval_examples_clean, device,
                            max_src_len=512, max_gen_len=32, batch_size=batch_size, debug=False
                        )
                        noisy_metrics = evaluate_tweetqa_generative(
                            model, tokenizer, noisy_examples, device,
                            max_src_len=512, max_gen_len=32, batch_size=batch_size, debug=False
                        )
                        
                        clean_em_list.append(clean_metrics['exact_match'])
                        clean_f1_list.append(clean_metrics['f1'])
                        print(f"Clean - EM: {clean_metrics['exact_match']:.2f}%, F1: {clean_metrics['f1']:.2f}%")
                        
                        noisy_em_list.append(noisy_metrics['exact_match'])
                        noisy_f1_list.append(noisy_metrics['f1'])
                        print(f"Noisy - EM: {noisy_metrics['exact_match']:.2f}%, F1: {noisy_metrics['f1']:.2f}%")
                    else:
                        print("TweetQA only supports generative models (LLaMA), skipping extractive model")
                        continue
                
                elif args.dataset == 'samsum':
                    # SAMSum ROUGE evaluation (batched for speed)
                    clean_metrics = evaluate_samsum_rouge(
                        model, tokenizer, eval_examples_clean, device,
                        max_src_len=512, max_gen_len=args.max_samsum_gen_len, 
                        batch_size=batch_size, debug=False
                    )
                    noisy_metrics = evaluate_samsum_rouge(
                        model, tokenizer, noisy_examples_list, device,
                        max_src_len=512, max_gen_len=args.max_samsum_gen_len, 
                        batch_size=batch_size, debug=False
                    )
                    
                    clean_r1_list.append(clean_metrics['rouge1'])
                    clean_r2_list.append(clean_metrics['rouge2'])
                    clean_rL_list.append(clean_metrics['rougeL'])
                    print(f"Clean - R1: {clean_metrics['rouge1']:.2f}%, R2: {clean_metrics['rouge2']:.2f}%, RL: {clean_metrics['rougeL']:.2f}%")
                    
                    noisy_r1_list.append(noisy_metrics['rouge1'])
                    noisy_r2_list.append(noisy_metrics['rouge2'])
                    noisy_rL_list.append(noisy_metrics['rougeL'])
                    print(f"Noisy - R1: {noisy_metrics['rouge1']:.2f}%, R2: {noisy_metrics['rouge2']:.2f}%, RL: {noisy_metrics['rougeL']:.2f}%")
                
                # Clear GPU memory
                del model
                torch.cuda.empty_cache()
            
            # Calculate and save statistics
            if args.dataset in ['squad', 'adversarial_qa', 'subjqa', 'tweetqa'] and len(clean_em_list) > 0:
                # Clean results
                clean_em_mean = np.mean(clean_em_list)
                clean_em_std = np.std(clean_em_list)
                clean_f1_mean = np.mean(clean_f1_list)
                clean_f1_std = np.std(clean_f1_list)
                
                # Noisy results
                noisy_em_mean = np.mean(noisy_em_list)
                noisy_em_std = np.std(noisy_em_list)
                noisy_f1_mean = np.mean(noisy_f1_list)
                noisy_f1_std = np.std(noisy_f1_list)
                
                results_map[f"{perturbation} - clean"] = {
                    'EM': f"{clean_em_mean:.2f}±{clean_em_std:.2f}",
                    'F1': f"{clean_f1_mean:.2f}±{clean_f1_std:.2f}"
                }
                
                results_map[f"{perturbation} - noisy"] = {
                    'EM': f"{noisy_em_mean:.2f}±{noisy_em_std:.2f}",
                    'F1': f"{noisy_f1_mean:.2f}±{noisy_f1_std:.2f}"
                }
                
                # Save raw results
                torch.save({
                    'clean_em': clean_em_list,
                    'clean_f1': clean_f1_list,
                    'noisy_em': noisy_em_list,
                    'noisy_f1': noisy_f1_list
                }, os.path.join(output_dir, f"noise_eval_{args.noise_type}_level{current_noise_level}.pt"))
            
            elif args.dataset == 'samsum' and len(clean_r1_list) > 0:
                # Clean results
                clean_r1_mean = np.mean(clean_r1_list)
                clean_r1_std = np.std(clean_r1_list)
                clean_r2_mean = np.mean(clean_r2_list)
                clean_r2_std = np.std(clean_r2_list)
                clean_rL_mean = np.mean(clean_rL_list)
                clean_rL_std = np.std(clean_rL_list)
                
                # Noisy results
                noisy_r1_mean = np.mean(noisy_r1_list)
                noisy_r1_std = np.std(noisy_r1_list)
                noisy_r2_mean = np.mean(noisy_r2_list)
                noisy_r2_std = np.std(noisy_r2_list)
                noisy_rL_mean = np.mean(noisy_rL_list)
                noisy_rL_std = np.std(noisy_rL_list)
                
                results_map[f"{perturbation} - clean"] = {
                    'R1': f"{clean_r1_mean:.2f}±{clean_r1_std:.2f}",
                    'R2': f"{clean_r2_mean:.2f}±{clean_r2_std:.2f}",
                    'RL': f"{clean_rL_mean:.2f}±{clean_rL_std:.2f}"
                }
                
                results_map[f"{perturbation} - noisy"] = {
                    'R1': f"{noisy_r1_mean:.2f}±{noisy_r1_std:.2f}",
                    'R2': f"{noisy_r2_mean:.2f}±{noisy_r2_std:.2f}",
                    'RL': f"{noisy_rL_mean:.2f}±{noisy_rL_std:.2f}"
                }
                
                # Save raw results
                torch.save({
                    'clean_r1': clean_r1_list,
                    'clean_r2': clean_r2_list,
                    'clean_rL': clean_rL_list,
                    'noisy_r1': noisy_r1_list,
                    'noisy_r2': noisy_r2_list,
                    'noisy_rL': noisy_rL_list
                }, os.path.join(output_dir, f"noise_eval_{args.noise_type}_level{current_noise_level}.pt"))
        
        # Print summary for this noise level
        print(f"\n{'='*50}")
        print(f"Results Summary - Noise Level {current_noise_level} - {args.noise_type} - model {args.model}")
        print(f"{'='*50}")
        for key, metrics in results_map.items():
            print(f"{key}:")
            for metric_name, metric_val in metrics.items():
                print(f"  {metric_name}: {metric_val}")