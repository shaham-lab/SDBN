import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)
from datasets import load_dataset
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

warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate DeBERTa-v3 LoRA model on SQuAD')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to saved LoRA model file (e.g., /path/to/model_0.pt)')
parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base',
                    help='Base model name used during training')
parser.add_argument('--lora_rank', type=int, default=4,
                    help='LoRA rank used during training')
parser.add_argument('--val_size', type=int, default=1000,
                    help='Number of validation examples to evaluate on')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed for reproducibility')
parser.add_argument('--batch_size', type=int, default=70,
                    help='Batch size for evaluation')
parser.add_argument('--device', type=int, default=0,
                    help='GPU device number')
parser.add_argument('--verbose', action='store_true',
                    help='Print detailed evaluation results')

args = parser.parse_args()

# Set random seed
utils.set_random_seed(args.seed)

# Set device
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("="*60)
print("🔍 EVALUATING DEBERTA LORA MODEL")
print("="*60)
print(f"Model path: {args.model_path}")
print(f"Base model: {args.model_name}")
print(f"LoRA rank: {args.lora_rank}")
print(f"Validation size: {args.val_size}")
print(f"Seed: {args.seed}")
print(f"Batch size: {args.batch_size}")

class SQuADDataset(Dataset):
    """Custom Dataset for SQuAD evaluation"""
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
        'f1': 100.0 * sum(f1_scores) / len(f1_scores),
        'total_questions': len(exact_scores)
    }

def evaluate_model(model, eval_dataloader, eval_dataset_for_model, eval_examples, tokenizer):
    """Evaluate model on dataset"""
    print("\n🔄 Running model evaluation...")
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
    
    print("🔄 Post-processing predictions...")
    predictions = postprocess_qa_predictions(
        all_start_logits, 
        all_end_logits, 
        eval_dataset_for_model, 
        eval_examples,
        tokenizer
    )
    
    # Create references
    references = {}
    for example in eval_examples:
        if len(example['answers']['text']) > 0:
            references[example['id']] = example['answers']['text'][0]
        else:
            references[example['id']] = ""
    
    print("🔄 Computing metrics...")
    metrics = compute_metrics(predictions, references)
    
    return metrics, predictions, references

def main():
    print("\n" + "="*60)
    print("🚀 STARTING EVALUATION")
    print("="*60)
    
    # Step 1: Load the model
    print("\n📂 Step 1: Loading LoRA model...")
    try:
        model = utils_gen.load_lora_gen_model(
            model_name=args.model_name,
            save_path=args.model_path,
            lora_rank=args.lora_rank,
            device=device
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Step 2: Load tokenizer
    print("\n📝 Step 2: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"✅ Tokenizer loaded: {args.model_name}")
    
    # Step 3: Load SQuAD validation dataset
    print("\n📊 Step 3: Loading SQuAD validation dataset...")
    dataset = load_dataset("squad")
    validation_examples = dataset["validation"]
    
    if args.val_size:
        validation_examples = validation_examples.select(range(min(args.val_size, len(validation_examples))))
    
    print(f"✅ Loaded {len(validation_examples)} validation examples")
    
    # Step 4: Prepare features
    print("\n🔧 Step 4: Preparing validation features...")
    validation_dataset = validation_examples.map(
        lambda x: prepare_validation_features(x, tokenizer),
        batched=True,
        remove_columns=validation_examples.column_names
    )
    
    eval_dataset_final = SQuADDataset({
        'input_ids': validation_dataset['input_ids'],
        'attention_mask': validation_dataset['attention_mask'],
        'example_id': validation_dataset['example_id']
    })
    
    eval_dataloader = DataLoader(eval_dataset_final, batch_size=args.batch_size)
    print(f"✅ Prepared {len(eval_dataset_final)} features in {len(eval_dataloader)} batches")
    
    # Step 5: Evaluate model
    print("\n🎯 Step 5: Evaluating model performance...")
    metrics, predictions, references = evaluate_model(
        model, eval_dataloader, validation_dataset, validation_examples, tokenizer
    )
    
    # Step 6: Print results
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"🎯 Exact Match (EM): {metrics['exact_match']:.2f}%")
    print(f"🎯 F1 Score: {metrics['f1']:.2f}%")
    print(f"📝 Total Questions: {metrics['total_questions']}")
    
    # Step 7: Show some examples if verbose
    if args.verbose:
        print("\n" + "="*60)
        print("📝 SAMPLE PREDICTIONS")
        print("="*60)
        
        sample_examples = validation_examples.select(range(min(5, len(validation_examples))))
        for i, example in enumerate(sample_examples):
            qid = example['id']
            question = example['question']
            context = example['context'][:200] + "..." if len(example['context']) > 200 else example['context']
            gold_answer = example['answers']['text'][0] if example['answers']['text'] else ""
            predicted_answer = predictions.get(qid, "")
            
            print(f"\n📝 Example {i+1}:")
            print(f"❓ Question: {question}")
            print(f"📄 Context: {context}")
            print(f"🎯 Gold Answer: '{gold_answer}'")
            print(f"🤖 Predicted: '{predicted_answer}'")
            print(f"✅ Match: {'Yes' if predicted_answer.lower().strip() == gold_answer.lower().strip() else 'No'}")
    
    # Step 8: Save results
    results_dict = {
        'model_path': args.model_path,
        'model_name': args.model_name,
        'lora_rank': args.lora_rank,
        'val_size': args.val_size,
        'seed': args.seed,
        'exact_match': metrics['exact_match'],
        'f1': metrics['f1'],
        'total_questions': metrics['total_questions']
    }
    
    # Save results to JSON
    results_file = args.model_path.replace('.pt', '_eval_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    
    # Final validation message
    if metrics['exact_match'] > 70 and metrics['f1'] > 75:
        print("🎉 Great! Your save-load functions work correctly - the model performance is good!")
    elif metrics['exact_match'] > 30 and metrics['f1'] > 40:
        print("⚠️  Model loaded but performance seems lower than expected. Check training.")
    else:
        print("❌ Low performance detected. There might be an issue with save-load functions.")

if __name__ == "__main__":
    main()
