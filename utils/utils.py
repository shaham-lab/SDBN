#utils.py
import torch
import random
import numpy as np
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType,LoraModel, get_peft_model_state_dict#,  AdapterConfig, PeftModel, PeftConfig
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
import adapters
from adapters.configuration import SeqBnConfig
from transformers import BertForSequenceClassification, BertConfig
from datasets import load_dataset, DatasetDict
import os
from peft import get_peft_model, LoraConfig, TaskType, set_peft_model_state_dict
import nltk
nltk.download('cmudict')  # download if you haven't already
from nltk.corpus import cmudict
import re
import spacy
device = None

# Banking77 dataset label mapping
TREC_LABEL_MAP = {
    "ABBR:abb": 0, "ABBR:exp": 1,
    "ENTY:animal": 2, "ENTY:body": 3, "ENTY:color": 4, "ENTY:cremat": 5,
    "ENTY:currency": 6, "ENTY:dismed": 7, "ENTY:event": 8, "ENTY:food": 9,
    "ENTY:instru": 10, "ENTY:lang": 11, "ENTY:letter": 12, "ENTY:other": 13,
    "ENTY:plant": 14, "ENTY:product": 15, "ENTY:religion": 16, "ENTY:sport": 17,
    "ENTY:substance": 18, "ENTY:symbol": 19, "ENTY:techmeth": 20, "ENTY:termeq": 21,
    "ENTY:veh": 22, "ENTY:word": 23,
    "DESC:def": 24, "DESC:desc": 25, "DESC:manner": 26, "DESC:reason": 27,
    "HUM:gr": 28, "HUM:ind": 29, "HUM:title": 30, "HUM:desc": 31,
    "LOC:city": 32, "LOC:country": 33, "LOC:mount": 34, "LOC:other": 35, "LOC:state": 36,
    "NUM:code": 37, "NUM:count": 38, "NUM:date": 39, "NUM:dist": 40, "NUM:money": 41,
    "NUM:ord": 42, "NUM:other": 43, "NUM:period": 44, "NUM:perc": 45,
    "NUM:speed": 46, "NUM:temp": 47, "NUM:volsize": 48, "NUM:weight": 49,
}

BANKING77_LABEL_MAP = {
    'activate_my_card': 0,
    'age_limit': 1,
    'apple_pay_or_google_pay': 2,
    'atm_support': 3,
    'automatic_top_up': 4,
    'balance_not_updated_after_bank_transfer': 5,
    'balance_not_updated_after_cheque_or_cash_deposit': 6,
    'beneficiary_not_allowed': 7,
    'cancel_transfer': 8,
    'card_about_to_expire': 9,
    'card_acceptance': 10,
    'card_arrival': 11,
    'card_delivery_estimate': 12,
    'card_linking': 13,
    'card_not_working': 14,
    'card_payment_fee_charged': 15,
    'card_payment_not_recognised': 16,
    'card_payment_wrong_exchange_rate': 17,
    'card_swallowed': 18,
    'cash_withdrawal_charge': 19,
    'cash_withdrawal_not_recognised': 20,
    'change_pin': 21,
    'compromised_card': 22,
    'contactless_not_working': 23,
    'country_support': 24,
    'declined_card_payment': 25,
    'declined_cash_withdrawal': 26,
    'declined_transfer': 27,
    'direct_debit_payment_not_recognised': 28,
    'disposable_card_limits': 29,
    'edit_personal_details': 30,
    'exchange_charge': 31,
    'exchange_rate': 32,
    'exchange_via_app': 33,
    'extra_charge_on_statement': 34,
    'failed_transfer': 35,
    'fiat_currency_support': 36,
    'get_disposable_virtual_card': 37,
    'get_physical_card': 38,
    'getting_spare_card': 39,
    'getting_virtual_card': 40,
    'lost_or_stolen_card': 41,
    'lost_or_stolen_phone': 42,
    'order_physical_card': 43,
    'passcode_forgotten': 44,
    'pending_card_payment': 45,
    'pending_cash_withdrawal': 46,
    'pending_top_up': 47,
    'pending_transfer': 48,
    'pin_blocked': 49,
    'receiving_money': 50,
    'Refund_not_showing_up': 51,
    'request_refund': 52,
    'reverted_card_payment?': 53,
    'supported_cards_and_currencies': 54,
    'terminate_account': 55,
    'top_up_by_bank_transfer_charge': 56,
    'top_up_by_card_charge': 57,
    'top_up_by_cash_or_cheque': 58,
    'top_up_failed': 59,
    'top_up_limits': 60,
    'top_up_reverted': 61,
    'topping_up_by_card': 62,
    'transaction_charged_twice': 63,
    'transfer_fee_charged': 64,
    'transfer_into_account': 65,
    'transfer_not_received_by_recipient': 66,
    'transfer_timing': 67,
    'unable_to_verify_identity': 68,
    'verify_my_identity': 69,
    'verify_source_of_funds': 70,
    'verify_top_up': 71,
    'virtual_card_not_working': 72,
    'visa_or_mastercard': 73,
    'why_verify_identity': 74,
    'wrong_amount_of_cash_received': 75,
    'wrong_exchange_rate_for_cash_withdrawal': 76
}

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed+1)
    torch.manual_seed(seed+2)
    torch.cuda.manual_seed(seed+3)
    torch.cuda.manual_seed_all(seed+4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class SST5Dataset(Dataset):
    def __init__(self, data):
        if isinstance(data, dict):
            data = pd.DataFrame(data)
            
        self.texts = data["text"].reset_index(drop=True)
        self.labels = data["label"].astype(int).reset_index(drop=True)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
    
    
def collate_fn(batch):
    # batch is a list of ((premise, hyp), label)
    pairs, labels = zip(*batch)
    # pairs is a tuple of (premise, hyp) tuples → convert to list
    return list(pairs), torch.tensor(labels, dtype=torch.long)

def get_embeddings(model, input_ids,alg=None):
    """
    Get embeddings from model based on input_ids.
    Works for both LoRA and regular BERT models.
    
    Args:
        model: BERT model (either with LoRA or without)
        input_ids: tensor of token ids
    Returns:
        embeddings tensor
    """
   
    try:
        # This is the most reliable way to get embeddings
        embedding_layer = model.get_input_embeddings()
        return embedding_layer(input_ids)
        
    except:
        if hasattr(model, 'base_model'):
            embedding_layer = model.base_model.get_input_embeddings()
            return embedding_layer(input_ids)
            
        else:
            embedding_layer = model.get_input_embeddings()
            return embedding_layer(input_ids)

def get_embedding_weights(model):
    """
    Get embedding weights from model. Works for both LoRA and regular BERT.
    
    Args:
        model: BERT model (either with LoRA or regular)
    Returns:
        embedding weight matrix
    """
    try:
        # For LoRA models
        if hasattr(model, 'base_model'):
            return model.base_model.get_input_embeddings().weight
        # For regular/BitFit models
        else:
            return model.get_input_embeddings().weight
    except:
        print("Model structure:", type(model))
        raise ValueError("Could not get embedding weights")
    
    
    
def count_parameters(model):
    """
    Count the number of parameters in the model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def perform_lora_adversarial_attack(optimizer, text, model, tokenizer, label, alg, epsilon=1e-4, pertubation='sdbn'):
    """
    Performs adversarial attack on the model's embedding space.
    Args:
        text: input text (list of strings)
        model: the fine-tuned model
        tokenizer: tokenizer
        label: tensor of labels
        epsilon: perturbation size
        pertubation: 'sdbn' (l-inf FGSM) or 'sdbn-h' (hybrid with char-level noise)
    Returns:
        adversarial_embeddings: perturbed embeddings tensor
    """
    model.eval()

    encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    embedding_weights = get_embedding_weights(model)
    embeddings = torch.nn.functional.embedding(input_ids, embedding_weights).clone().detach().requires_grad_(True).to(device)

    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    logits = outputs.logits
    probs = F.log_softmax(logits, dim=-1)

    label = label.to(device)
    loss_scalar = -torch.gather(probs, 1, label.unsqueeze(dim=-1)).sum()
    loss_scalar.backward()

    optimizer.zero_grad()

    if pertubation == 'sdbn':
        adversarial_embeddings = (embeddings + epsilon * torch.sign(embeddings.grad)).detach()
    elif pertubation == 'sdbn-h':
        # Hybrid: (1 - 5%) sdbn + 5% gradient-guided character-level noise
        batch_size = embeddings.shape[0]
        num_hotflip = max(1, int(0.05 * batch_size))
        num_sdbn = batch_size - num_hotflip

        sdbn_delta = torch.sign(embeddings.grad[:num_sdbn]) * epsilon
        adv_embeds = embeddings.clone().detach()
        adv_embeds[:num_sdbn] = embeddings[:num_sdbn] + sdbn_delta

        for i in range(num_sdbn, batch_size):
            char_noise_func = random.choice([
                delete_char_i, swap_char_i, double_char_i, phonetic_char_i,
                insert_char_after_i, cyrillic_char_i, random_capitalization_i
            ])
            orig_text = text[i]
            orig_embed = embeddings[i].detach()
            grad = embeddings.grad[i].detach()
            best_score = None
            best_embed = orig_embed
            for j in range(len(orig_text)):
                x_prime = char_noise_func(orig_text, j)
                encoded_prime = tokenizer(x_prime, return_tensors='pt', padding=True, truncation=True, max_length=128)
                input_ids_prime = encoded_prime['input_ids'].to(device)
                embed_prime = torch.nn.functional.embedding(input_ids_prime, embedding_weights).squeeze(0)
                if embed_prime.shape[0] > orig_embed.shape[0]:
                    embed_prime = embed_prime[:orig_embed.shape[0]]
                elif embed_prime.shape[0] < orig_embed.shape[0]:
                    pad = torch.zeros(orig_embed.shape[0] - embed_prime.shape[0], embed_prime.shape[1], device=device)
                    embed_prime = torch.cat([embed_prime, pad], dim=0)
                score = torch.sum(grad * (embed_prime - orig_embed)).item()
                if best_score is None or score > best_score:
                    best_score = score
                    best_embed = embed_prime
            adv_embeds[i] = best_embed
        adversarial_embeddings = adv_embeds
    else:
        raise ValueError(f"Unknown pertubation: {pertubation}")

    return adversarial_embeddings




def train_one_epoch(model, train_loader, test_loader, tokenizer, optimizer, device,
    adv_percent=0, checkpointing_enable=False, alg=None, epsilon=1e-4,
    pertubation='sdbn'):
    """
    Train for one epoch with optional adversarial training, then evaluate.
    Returns:
        train_loss, test_loss, train_acc, test_acc, model
    """
    model.train()
    train_loss = 0
    test_loss = 0
    train_correct = 0
    train_total = 0

    if checkpointing_enable:
        model.gradient_checkpointing_enable()

    for batch_idx, (texts, labels) in enumerate(train_loader):
        torch.cuda.empty_cache()
        use_adversarial = torch.rand(1).item() < adv_percent

        if use_adversarial:
            adv_embedding = perform_lora_adversarial_attack(
                optimizer, texts, model, tokenizer, labels,
                alg, epsilon=epsilon, pertubation=pertubation
            )

        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        optimizer.zero_grad()

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = labels.clone().detach().to(device)

        embeddings = adv_embedding if use_adversarial else get_embeddings(model, input_ids, alg)

        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        train_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            current_acc = 100. * train_correct / train_total
            print(f'Train Batch [{batch_idx + 1}/{len(train_loader)}] - '
                  f'Loss: {loss.item():.4f} - Accuracy: {current_acc:.2f}%')

    train_loss = train_loss / len(train_loader)
    train_acc = 100. * train_correct / train_total
    torch.cuda.empty_cache()

    # Evaluation phase
    model.eval()
    test_correct = 0
    test_total = 0

    print("\nStarting evaluation...")
    with torch.no_grad():
        for batch_idx, (texts, labels) in enumerate(test_loader):
            torch.cuda.empty_cache()
            inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = labels.clone().detach().to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            test_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                current_acc = 100. * test_correct / test_total
                print(f'Test Batch [{batch_idx + 1}/{len(test_loader)}] - Accuracy: {current_acc:.2f}%')

    test_acc = 100. * test_correct / test_total
    test_loss = test_loss / len(test_loader)

    print(f'\nEpoch Summary:')
    print(f'Training Loss: {train_loss:.4f}')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Training Accuracy: {train_acc:.2f}%')
    print(f'Test Accuracy: {test_acc:.2f}%')

    torch.cuda.empty_cache()
    return train_loss, test_loss, train_acc, test_acc, model
def evaluate_model(model, test_loader, tokenizer):
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (texts, labels) in enumerate(test_loader):
            # Tokenize batch
            inputs = tokenizer(
                texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=128
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            # The above code is not valid Python code. It seems to contain some text "labels" and "
            labels = labels.clone().detach().to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels 
            )
            logits = outputs.logits
            loss = outputs.loss
            test_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            torch.cuda.empty_cache()
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                current_acc = 100. * test_correct / test_total
                print(f'Test Batch [{batch_idx + 1}/{len(test_loader)}] - '
                      f'Accuracy: {current_acc:.2f}%')

    # Calculate final test accuracy
    test_acc = 100. * test_correct / test_total
    test_loss = test_loss / len(test_loader)
    
    
    return test_loss, test_acc

def get_preds(model, test_loader, tokenizer):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (texts, labels) in enumerate(test_loader):
            # Tokenize batch
            inputs = tokenizer(
                texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=128
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = labels.clone().detach().to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f'Test Batch [{batch_idx + 1}/{len(test_loader)}] - Processing...')

    # Convert to numpy arrays to ensure same dimensions as test_loader labels
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return all_preds

def prepare_model_for_lora(num_classes, model_name="bert-base-uncased", lora_rank=12):
    """
    Prepare the BERT model with LoRA configuration
    """
    # Load base model and determine target modules and classifier attribute
    if "deberta" in model_name.lower():
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
        target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
        classifier_attr = "classifier"  # DeBERTa classifier is directly on model
    elif "bert" in model_name.lower():
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
        target_modules = ["query", "key", "value", "dense"]
        classifier_attr = "base_model.classifier"  # BERT's classifier is under base_model
    else:
        raise ValueError("Unsupported model type")

    
    

    lora_config = LoraConfig(
        r=lora_rank,                     # Increased rank
        lora_alpha=64,           
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        inference_mode=False      # Changed to False for training
    )
    
    # Create PEFT model
    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        if 'lora' in name or 'classifier' in name:  # LoRA parameters should be trainable
            param.requires_grad = True
        else:  # Other parameters should be frozen
            param.requires_grad = False
            
    model.config._classifier_attr = classifier_attr

    model.print_trainable_parameters()  # This will show us if LoRA parameters are actually trainable
    
    return model

def prepare_model_for_qlora(num_classes, model_name="bert-base-uncased", lora_rank=12):
    """
    Prepare the model with QLoRA configuration
    """
    from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    import torch
    
    # Set up quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,                # Load model in 4-bit precision
        bnb_4bit_quant_type="nf4",        # Normalized float 4 format
        bnb_4bit_compute_dtype=torch.float16,  # Compute in float16
        bnb_4bit_use_double_quant=True    # Use nested quantization for memory efficiency
    )
    
    # Load base model with quantization
    if 'deberta' in model_name:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            quantization_config=quantization_config
        )
    elif 'bert' in model_name:
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            quantization_config=quantization_config
        )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Determine target modules based on model type
    if 'deberta' in model_name.lower():
        target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
        classifier_attr = "classifier"  # DeBERTa: classifier is directly on the model
    else:
        target_modules = ["query", "key", "value", "dense"]
        classifier_attr = "base_model.classifier"
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=lora_rank,                     # Rank
        lora_alpha=64,                   # Alpha parameter
        target_modules=target_modules,   # Which modules to apply LoRA to
        lora_dropout=0.1,                # Dropout probability for LoRA
        bias="none",                     # Bias type
        task_type=TaskType.SEQ_CLS,      # Task type
        inference_mode=False             # Training mode
    )
    
    # Create PEFT model
    model = get_peft_model(model, lora_config)
    
    for name, param in model.named_parameters():
        if 'lora' in name or classifier_attr in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.print_trainable_parameters()

    # Store classifier attribute and QLoRA flag in model.config for later reference in saving/loading.
    model.config._classifier_attr = classifier_attr
    model.config.qlora = True  # We are using QLoRA in this setup

    return model
    
    return model

def prepare_model_for_adapter(num_classes, model_name="bert-base-uncased", bottle_neck=16):

    # Load base model
    
    is_deberta = "deberta" in model_name.lower()
    
    # Load base model based on model type
    if is_deberta:
        if "v3" in model_name:
            from adapters import DebertaV2AdapterModel
            
            model = DebertaV2AdapterModel.from_pretrained(model_name)
        else:
            from adapters import DebertaAdapterModel
            model = DebertaAdapterModel.from_pretrained(model_name)
    else:
        model = adapters.BertAdapterModel.from_pretrained(model_name)
    
    hidden_size = model.config.hidden_size
    
    reduction_factor = int(hidden_size/bottle_neck)
    print(">>>hidden_size:",hidden_size," reduction_factor:",reduction_factor)
    
    adapter_config = SeqBnConfig(
    mh_adapter=False,
    output_adapter=True,
    reduction_factor=reduction_factor,
    non_linearity="relu",
    original_ln_before=True,
    original_ln_after=True,
    ln_before=False,
    ln_after=False,
    is_parallel=False,
    )
    
    adapter_name = "task_adapter"
    model.add_adapter(adapter_name, config=adapter_config)

    #
    # 3. Add a classification head named "classifier" (8 labels).
    #
    model.add_classification_head("classifier", num_labels=num_classes)

    #
    # 4. Activate the adapter + classification head.
    #
    model.set_active_adapters(adapter_name)  # use our "task_adapter"
    model.active_head = "classifier"         # use the "classifier" head

    # Freeze the base BERT parameters; only the adapter + classifier head train
    model.train_adapter(adapter_name)
    # Create PEFT model
    
    trainable_params = []  
    
    for name, param in model.named_parameters():
        if 'classifier' in name or 'adapter_down' in name or 'adapter_up' in name:  # Keep classifier trainable
            trainable_params.append(name)
            param.requires_grad = True
        else:                     # Freeze everything else
            param.requires_grad = False
            
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params} / {total_params} ({(trainable_params/total_params)*100:.2f}%)")
    print("Traonable params:")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)
    return model


def save_model_adapter(model, output_dir, seed, dataset, mode, epoch="final"):
    
    weights_path = os.path.join(output_dir, f"weights_baseline_{mode}_seed{seed}_epoch_{epoch}.pt")
    #
    # 1. Collect adapter parameters
    #
    # The adapter modules are stored in each Transformer layer under
    # e.g. `bert.encoder.layer.X.output.adapters.{adapter_name}` 
    # or `model.encoder.layer.X.output.adapters.{adapter_name}` etc.
    # We'll filter the full named_params by checking if `adapter_name` is in the parameter key.
    #
    adapter_state_dict = {}
    for param_name, param_tensor in model.named_parameters():
        if "task_adapter" in param_name:
            adapter_state_dict[param_name] = param_tensor.cpu().clone()

    #
    # 2. Collect the classification head parameters
    #
    # If you created the head with:
    #     model.add_classification_head("classifier", num_labels=8)
    # it should live in `model.heads["classifier"]`.
    # We'll just grab its state_dict().
    #
    classifier_state_dict = model.heads["classifier"].state_dict()

    #
    # 3. Combine into one dictionary
    #
    full_state_dict = {
        "adapter": adapter_state_dict,
        "classifier": classifier_state_dict
    }

    # 4. Save to disk
    torch.save(full_state_dict, weights_path)
    print(f"Model saved to {weights_path}")
    
    
def load_adapter_model(model_path, number_of_labels, model_name="bert-base-uncased", bottle_neck=16):
    """
    Load a saved adapter model.
    
    Args:
        model_path (str): Path to the saved model weights
        number_of_labels (int): Number of classification labels
        model_name (str): Base model name (default: "bert-base-uncased")
        bottle_neck (int): Bottleneck size for adapter (default: 16)
    """
    try:
        # First prepare the model with adapter
        model = prepare_model_for_adapter(number_of_labels, model_name, bottle_neck)
        
        # Load the saved weights
        #saved_weights = torch.load(model_path)
        saved_weights = torch.load(model_path, map_location=device)

        
        # Load adapter weights
        adapter_weights = saved_weights['adapter']
        for name, param in model.named_parameters():
            if name in adapter_weights:
                param.data = adapter_weights[name].to(param.device)
        
        # Load classifier weights
        classifier_weights = saved_weights['classifier']
        model.heads["classifier"].load_state_dict(classifier_weights)
        
        # Make sure adapter and classifier are active
        model.set_active_adapters("task_adapter")
        model.active_head = "classifier"
        
        return model
        
    except Exception as e:
        print(f"Error loading model weights from {model_path}: {str(e)}")
        raise e
    



def save_model_lora(model, output_dir, seed, dataset, mode, epoch="final"):
    
    weights_path = os.path.join(output_dir, f"weights_baseline_{mode}_seed{seed}_epoch_{epoch}.pt")
    # Determine classifier state based on the stored attribute
    classifier_attr = model.config._classifier_attr
    if classifier_attr == "base_model.classifier":
        classifier_state = model.base_model.classifier.state_dict()
    elif classifier_attr == "classifier":
        classifier_state = model.classifier.state_dict()
    else:
        raise ValueError("Unknown classifier attribute")
    

    full_state_dict = {
        "lora": get_peft_model_state_dict(model),
        "classifier": classifier_state
    }
    torch.save(full_state_dict, weights_path)
    
    print(f"Model saved to {weights_path}")
    
    return weights_path
    
def get_qlora_state_dict(model):
    """
    Retrieve QLoRA-specific state.
    Replace this logic with the actual state retrieval for your QLoRA implementation.
    """
    if hasattr(model, "qlora_module"):
        return model.qlora_module.state_dict()
    return {}

def set_qlora_state_dict(model, qlora_state):
    """
    Restore QLoRA-specific state.
    Replace this logic with the actual state restoration for your QLoRA implementation.
    """
    if hasattr(model, "qlora_module"):
        model.qlora_module.load_state_dict(qlora_state)
def save_model_qlora(model, output_dir, seed, dataset, mode, epoch="final"):
    """
    Save the QLoRA model checkpoint. This saves the LoRA (QLoRA) state,
    the classifier state (and, if applicable, additional QLoRA-specific state).
    """
    import os
    # Determine classifier state based on stored attribute.
    classifier_attr = model.config._classifier_attr
    if classifier_attr == "base_model.classifier":
        classifier_state = model.base_model.classifier.state_dict()
    elif classifier_attr == "classifier":
        classifier_state = model.classifier.state_dict()
    else:
        raise ValueError("Classifier attribute not found in the model.")

    weights_path = os.path.join(output_dir, f"weights_baseline_{mode}_seed{seed}_epoch_{epoch}.pt")
    full_state_dict = {
        "lora": get_peft_model_state_dict(model),
        "classifier": classifier_state,
        "qlora": model.config.qlora  # Save the QLoRA flag
    }
    # If QLoRA-specific state exists, include it.
    if model.config.qlora:
        full_state_dict["qlora_state"] = get_qlora_state_dict(model)

    import torch
    torch.save(full_state_dict, weights_path)
    print(f"Model saved to {weights_path}")
def load_qlora_model(model_path, number_of_labels, model_name="bert-base-uncased", rank=12):
    """
    Load the QLoRA model checkpoint. This builds the model with the correct configuration,
    then loads the LoRA, classifier, and QLoRA-specific states.
    """
    import torch
    # Build the model using our QLoRA preparation function.
    model = prepare_model_for_qlora(num_classes=number_of_labels, model_name=model_name, lora_rank=rank)
    state_dict = torch.load(model_path, map_location="cpu")  # or map_location=device if defined

    # Load LoRA (QLoRA) state.
    set_peft_model_state_dict(model, state_dict["lora"])

    # Load classifier state.
    classifier_attr = model.config._classifier_attr
    if classifier_attr == "base_model.classifier":
        model.base_model.classifier.load_state_dict(state_dict["classifier"])
    elif classifier_attr == "classifier":
        model.classifier.load_state_dict(state_dict["classifier"])
    else:
        raise ValueError("Classifier attribute not found in the model.")

    # If QLoRA is used and a QLoRA state was saved, restore it.
    if model.config.qlora and "qlora_state" in state_dict:
        set_qlora_state_dict(model, state_dict["qlora_state"])

    return model

def load_lora_model(model_path, number_of_labels, model_name="bert-base-uncased", rank=12):
    model = prepare_model_for_lora(num_classes=number_of_labels, model_name=model_name, lora_rank=rank)
    state_dict = torch.load(model_path, map_location=device)
    set_peft_model_state_dict(model, state_dict["lora"])
    classifier_attr = model.config._classifier_attr
    if classifier_attr == "base_model.classifier":
        model.base_model.classifier.load_state_dict(state_dict["classifier"])
    elif classifier_attr == "classifier":
        model.classifier.load_state_dict(state_dict["classifier"])
    else:
        raise ValueError("Unknown classifier attribute")
    return model

def set_classifier_state_dict(model, state_dict):
    if hasattr(model, "base_model") and hasattr(model.base_model, "classifier"):
        model.base_model.classifier.load_state_dict(state_dict)
    elif hasattr(model, "classifier"):
        model.classifier.load_state_dict(state_dict)
    else:
        raise ValueError("Classifier attribute not found in the model.")
        exit(1)
    
def get_classifier_state_dict(model):
    # For BERT, classifier is typically under base_model
    if hasattr(model, "base_model") and hasattr(model.base_model, "classifier"):
        return model.base_model.classifier.state_dict()
    # For DeBERTa, classifier is directly on the model
    elif hasattr(model, "classifier"):
        return model.classifier.state_dict()
    else:
        raise ValueError("Classifier attribute not found in the model.")
    
def get_classifier_and_pooler_state_dict(model):
    # For DeBERTa, the classifier is directly on the model and there is a pooler.
    if hasattr(model, "pooler"):
        return {
            "classifier": model.classifier.state_dict(),
            "pooler": model.pooler.state_dict()
        }
    # For BERT, if using BertForSequenceClassification, classifier is directly available.
    elif hasattr(model, "classifier"):
        return {"classifier": model.classifier.state_dict()}
    else:
        raise ValueError("Classifier attribute not found in the model.")

def set_classifier_and_pooler_state_dict(model, state_dict):
    if hasattr(model, "pooler"):
        model.classifier.load_state_dict(state_dict["classifier"])
        model.pooler.load_state_dict(state_dict["pooler"])
    elif hasattr(model, "classifier"):
        model.classifier.load_state_dict(state_dict["classifier"])
    else:
        raise ValueError("Classifier attribute not found in the model.")
    
def prepare_model_for_bitfit(num_classes, model_name="bert-base-uncased"):
    
    if 'deberta' in model_name:
        model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes  # Set this to your number of classes
        )
    elif 'bert' in model_name:
        model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes
        )
 
    trainable_params = []        
    for name, param in model.named_parameters():
        if "bias" in name or "classifier" in name:
            param.requires_grad = True
            trainable_params.append(name)
        else:
            param.requires_grad = False
    
    # Handle embeddings differently for different model types
    if 'deberta' in model_name:
        if 'v3' in model_name:
            # DeBERTa v3 uses 'deberta' rather than 'bert' prefix
            for param in model.deberta.embeddings.parameters():
                param.requires_grad = False
        else:
            # Earlier DeBERTa versions
            for param in model.deberta.embeddings.parameters():
                param.requires_grad = False
    else:
        # BERT models
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
            
    

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params} / {total_params} ({(trainable_params/total_params)*100:.2f}%)")
    
    return model


def prepare_model_for_full_ft(num_classes, model_name="bert-base-uncased"):
    """Prepare model for full fine-tuning (all parameters trainable)."""
    if 'deberta' in model_name:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
    elif 'bert' in model_name:
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )

    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params} / {total_params} ({(trainable_params/total_params)*100:.2f}%)")

    return model


def save_model_full_ft(model, output_dir, seed, dataset, mode, epoch="final"):
    weights_path = os.path.join(output_dir, f"weights_full_ft_{mode}_seed{seed}_epoch_{epoch}.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Model saved to {weights_path}")


def save_model_bitfit(model, output_dir, seed, dataset, mode, epoch="final"):
    
    weights_path = os.path.join(output_dir, f"weights_bitfit_{mode}_seed{seed}_epoch_{epoch}.pt")
    torch.save(model.state_dict(), weights_path)	
 
    
    print(f"Model saved to {weights_path}")
    
    
def load_bitfit_model(model_path, number_of_labels, model_name="bert-base-uncased"):
    """
    Load a saved BitFit model.
    
    Args:
        model_path (str): Path to the saved model weights
        number_of_labels (int): Number of classification labels
        model_name (str): Base model name (default: "bert-base-uncased")
    """
    try:
        # First prepare the BitFit model
        model = prepare_model_for_bitfit(number_of_labels, model_name)
        
        # Load the saved weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        # Ensure the same parameters are trainable as in prepare_model_for_bitfit
        for name, param in model.named_parameters():
            if "bias" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        return model
        
    except Exception as e:
        print(f"Error loading model weights from {model_path}: {str(e)}")
        raise e
    
def load_full_ft_model(model_path, number_of_labels, model_name="bert-base-uncased"):
    """
    Load a saved full fine-tuning model.

    Args:
        model_path (str): Path to the saved model weights
        number_of_labels (int): Number of classification labels
        model_name (str): Base model name
    """
    try:
        model = prepare_model_for_full_ft(number_of_labels, model_name)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        return model
    except Exception as e:
        print(f"Error loading model weights from {model_path}: {str(e)}")
        raise e


def get_ArSarcasm_ds(drop_msa=False, version='v2', ignore_list_dialects=[]):
    if version == 'v2':
        dir = os.path.join('ArSarcasm-v2', 'ArSarcasm-v2')
    elif version == 'v1':
        dir = os.path.join("ArSarcasm", "dataset")
    else:
        print("ERROR: ArSarcasm Version is invalid:", version)
        exit(1)
    train_set = pd.read_csv(os.path.join(dir, "ArSarcasm_train.csv"))
    test_set = pd.read_csv(os.path.join(dir, "ArSarcasm_test.csv"))
    
    # adapt labels to numbers
    all_dialects = ['levant',
                    'magreb',
                    'msa',
                    'gulf',
                    'egypt']
    

    if drop_msa:
        # Define dialect map without 'msa' and with continuous indices
        dialect_map = {'levant': 0, 'magreb': 1, 'gulf': 2, 'egypt': 3}
        
        # Filter out 'msa' dialect from both train and test sets
        train_set = train_set[train_set['dialect'] != 'msa']
        test_set = test_set[test_set['dialect'] != 'msa']
    else:
        dialect_map = {'levant': 0, 'magreb': 1, 'msa': 2, 'gulf': 3, 'egypt': 4}
    

    # rmove all dialect from ignore_list_dialects
    for dialect in ignore_list_dialects:
        train_set = train_set[train_set['dialect'] != dialect]
        test_set = test_set[test_set['dialect'] != dialect]
        
    # Map dialects to numbers
    train_set['dialect'] = train_set['dialect'].map(dialect_map)
    test_set['dialect'] = test_set['dialect'].map(dialect_map)
    
    sarcasm_map = {False: 0, True: 1}
    train_set['sarcasm'] = train_set['sarcasm'].map(sarcasm_map)
    test_set['sarcasm'] = test_set['sarcasm'].map(sarcasm_map)
    
    if version == 'v1':
        original_sentiment_map = {'negative':0, 'neutral':1, 'positive':2}
        train_set['original_sentiment'] = train_set['original_sentiment'].map(original_sentiment_map)
        test_set['original_sentiment'] = test_set['original_sentiment'].map(original_sentiment_map)
    elif version == 'v2':
        sentiment_map = {'NEG':0, 'NEU':1, 'POS':2}
        train_set['sentiment'] = train_set['sentiment'].map(sentiment_map)
        test_set['sentiment'] = test_set['sentiment'].map(sentiment_map)
        

    print("Dialects Map:", dialect_map)     
    return train_set, test_set
    
    
def get_ArSarcasm_testloader_sentiment_perdialect(dialect, batch_size):
    all_dialects = ['levant', 'magreb', 'gulf', 'egypt', 'msa']
    all_dialects.remove(dialect)
    train, test = get_ArSarcasm_ds(drop_msa=False,
                                        ignore_list_dialects=all_dialects)
    
    dataset = DatasetDict({'text': test['tweet'], 'label' : test['sentiment']})
    
    #test_set = SST5Dataset(pd.DataFrame(dataset['test']))
    
    #loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=True)
    
    return dataset

map_bank = {
    0: "activate_my_card",
    1: "age_limit",
    2: "apple_pay_or_google_pay",
    3: "atm_support",
    4: "automatic_top_up",
    5: "balance_not_updated_after_bank_transfer",
    6: "balance_not_updated_after_cheque_or_cash_deposit",
    7: "beneficiary_not_allowed",
    8: "cancel_transfer",
    9: "card_about_to_expire",
    10: "card_acceptance",
    11: "card_arrival",
    12: "card_delivery_estimate",
    13: "card_linking",
    14: "card_not_working",
    15: "card_payment_fee_charged",
    16: "card_payment_not_recognised",
    17: "card_payment_wrong_exchange_rate",
    18: "card_swallowed",
    19: "cash_withdrawal_charge",
    20: "cash_withdrawal_not_recognised",
    21: "change_pin",
    22: "compromised_card",
    23: "contactless_not_working",
    24: "country_support",
    25: "declined_card_payment",
    26: "declined_cash_withdrawal",
    27: "declined_transfer",
    28: "direct_debit_payment_not_recognised",
    29: "disposable_card_limits",
    30: "edit_personal_details",
    31: "exchange_charge",
    32: "exchange_rate",
    33: "exchange_via_app",
    34: "extra_charge_on_statement",
    35: "failed_transfer",
    36: "fiat_currency_support",
    37: "get_disposable_virtual_card",
    38: "get_physical_card",
    39: "getting_spare_card",
    40: "getting_virtual_card",
    41: "lost_or_stolen_card",
    42: "lost_or_stolen_phone",
    43: "order_physical_card",
    44: "passcode_forgotten",
    45: "pending_card_payment",
    46: "pending_cash_withdrawal",
    47: "pending_top_up",
    48: "pending_transfer",
    49: "pin_blocked",
    50: "receiving_money",
    51: "Refund_not_showing_up",
    52: "request_refund",
    53: "reverted_card_payment?",
    54: "supported_cards_and_currencies",
    55: "terminate_account",
    56: "top_up_by_bank_transfer_charge",
    57: "top_up_by_card_charge",
    58: "top_up_by_cash_or_cheque",
    59: "top_up_failed",
    60: "top_up_limits",
    61: "top_up_reverted",
    62: "topping_up_by_card",
    63: "transaction_charged_twice",
    64: "transfer_fee_charged",
    65: "transfer_into_account",
    66: "transfer_not_received_by_recipient",
    67: "transfer_timing",
    68: "unable_to_verify_identity",
    69: "verify_my_identity",
    70: "verify_source_of_funds",
    71: "verify_top_up",
    72: "virtual_card_not_working",
    73: "visa_or_mastercard",
    74: "why_verify_identity",
    75: "wrong_amount_of_cash_received",
    76: "wrong_exchange_rate_for_cash_withdrawal"
}



# noise defs

def delete_random_word(sentence):
    # Split the sentence into words
    words = sentence.split()
    # If there are no words or just one word, return the sentence as is
    if len(words) <= 1:
        return sentence
    # Choose a random index
    random_index = random.randrange(len(words))
    # Remove the word at the random index
    del words[random_index]
    # Join the remaining words back into a sentence
    return " ".join(words)

def swap_two_random_words(sentence):
    # Split the sentence into words
    words = sentence.split()
    # If there are less than two words, return the sentence as is
    if len(words) < 2:
        return sentence
    # Randomly choose two distinct indices
    idx1, idx2 = random.sample(range(len(words)), 2)
    # Swap the words at the selected indices
    words[idx1], words[idx2] = words[idx2], words[idx1]
    # Join the words back into a sentence
    return " ".join(words)

# homophone noise
cmu = cmudict.dict()  # word → list of phoneme sequences

# 1. invert to phoneme→words
phones_to_words = {}
for word, pron_list in cmu.items():
    for pron in pron_list:
        key = tuple(pron)
        phones_to_words.setdefault(key, []).append(word)

# 2. homophone lookup
def get_homophones(word):
    """Return a list of words that are pronounced exactly like `word`."""
    word_l = word.lower()
    if word_l not in cmu:
        return []
    homos = set()
    for pron in cmu[word_l]:
        homos.update(phones_to_words.get(tuple(pron), []))
    homos.discard(word_l)
    return list(homos)

# 5. sentence‐level homophone noise
def noise_homophones(sentence, p=0.3):
    """
    With probability p for each word, replace it with a random homophone (if any).
    """
    tokens = sentence.split()
    out = []
    for tok in tokens:
        if random.random() < p:
            cands = get_homophones(tok)
            if cands:
                out.append(random.choice(cands))
                continue
        out.append(tok)
    return " ".join(out)


CONTRACTION_MAP = {
    # ---------- BE / IS / ARE ----------
    "I am": "I'm",
    "you are": "you're",
    "we are": "we're",
    "they are": "they're",
    "he is": "he's",
    "she is": "she's",
    "it is": "it's",
    "this is": "this's",
    "that is": "that's",
    "there is": "there's",
    "here is": "here's",
    "where is": "where's",
    "who is": "who's",
    "what is": "what's",
    "when is": "when's",
    "why is": "why's",
    "how is": "how's",
    # regional / negative catch‑all
    "am not": "ain't",
    "is not": "isn't",
    "are not": "aren't",
    "was not": "wasn't",
    "were not": "weren't",
    # ---------- HAVE / HAS / HAD ----------
    "I have": "I've",
    "you have": "you've",
    "we have": "we've",
    "they have": "they've",
    "he has": "he's",
    "she has": "she's",
    "it has": "it's",
    "who has": "who's",
    "could have": "could've",
    "should have": "should've",
    "would have": "would've",
    "must have": "must've",
    "might have": "might've",
    "may have": "may've",
    # negative
    "have not": "haven't",
    "has not": "hasn't",
    "had not": "hadn't",
    # ---------- WILL / WOULD ----------
    "I will": "I'll",
    "you will": "you'll",
    "he will": "he'll",
    "she will": "she'll",
    "it will": "it'll",
    "we will": "we'll",
    "they will": "they'll",
    "that will": "that'll",
    "there will": "there'll",
    "this will": "this'll",
    "I would": "I'd",
    "you would": "you'd",
    "he would": "he'd",
    "she would": "she'd",
    "we would": "we'd",
    "they would": "they'd",
    "there would": "there'd",
    "that would": "that'd",
    # negatives / irregular
    "will not": "won't",
    "would not": "wouldn't",
    "shall not": "shan't",
    # ---------- DO / DID / DOES ----------
    "do not": "don't",
    "does not": "doesn't",
    "did not": "didn't",
    "cannot": "can't",
    "can not": "can't",
    "could not": "couldn't",
    "should not": "shouldn't",
    "must not": "mustn't",
    "might not": "mightn't",
    "need not": "needn't",
    # ---------- LET / US ----------
    "let us": "let's",
    # ---------- GOING / WANT / GOT ----------
    "going to": "gonna",
    "want to": "wanna",
    "got to": "gotta",
    "got you": "gotcha",
    "give me": "gimme",
    "let me": "lemme",
    "tell them": "tell 'em",
    "of the": "o' the",
    # ---------- PRONOUN+HAVE informal ----------
    "ya all": "y'all",
    "you all": "y'all",
    # ---------- TIME SHORTENINGS ----------
    "until": "'til",
    "because": "'cause",
    # ---------- MISC ----------
    "kind of": "kinda",
    "sort of": "sorta",
    "out of": "outta",
    "more of": "more o'",
    
    # ---------- SMS/CHAT SPECIFIC ADDITIONS ----------
    # Common letter/digit substitutions
    "for": "4",
    "to": "2",
    "too": "2",
    "two": "2",
    "you": "u",
    "your": "ur",
    "are": "r",
    "see": "c",
    "why": "y",
    "be": "b",
    "before": "b4",
    "thanks": "thx",
    "please": "pls",
    "okay": "ok",
    
    # Word shortenings/abbreviations
    "about": "abt",
    "with": "w/",
    "without": "w/o",
    "what are you": "wru",
    "as far as I know": "afaik",
    "as soon as possible": "asap",
    "by the way": "btw",
    "for your information": "fyi",
    "in my opinion": "imo",
    "in my humble opinion": "imho",
    "oh my god": "omg",
    "rolling on floor laughing": "rofl",
    "laughing out loud": "lol",
    "talk to you later": "ttyl",
    "be right back": "brb",
    "by the way": "btw",
    "laugh my ass off": "lmao",
    "on my way": "omw",
    "shaking my head": "smh",
    
    # Common informal reductions
    "probably": "prob",
    "definitely": "def",
    "whatever": "whatev",
    "something": "somethin",
    "nothing": "nothin",
    "going": "goin",
    "coming": "comin",
    "talking": "talkin",
    "having": "havin",
    "doing": "doin",
    "getting": "gettin",
    "trying": "tryin",
    "alright": "aight",
    "tonight": "2nite",
    "tomorrow": "tmrw",
    "today": "2day",
    
    # Common phrase shortenings
    "I don't know": "idk",
    "I don't care": "idc",
    "no problem": "np",
    "thank you": "ty",
    "thank you very much": "tyvm",
    "love you": "ly",
    "love you too": "ly2",
    "see you": "cu",
    "see you later": "cul8r",
    "great": "gr8",
    "wait": "w8",
    "message": "msg",
    "pictures": "pics",
    "please let me know": "plmk",
    "talk to you": "tty",
    "just kidding": "jk",
    "just saying": "js",
    "to be honest": "tbh",
    
    # Emotional/reaction shortenings
    "congratulations": "congrats",
    "excited": "xcited",
    "seriously": "srsly",
    "not going to lie": "ngl",
    "no big deal": "nbd",
    
    # Drop vowels (common in texting)
    "people": "ppl",
    "between": "btwn",
    "would": "wld",
    "could": "cld",
    "should": "shld",
    "weekend": "wknd",
    "message": "msg",
    
    # G-dropping (common in casual typing)
    "looking": "lookin",
    "checking": "checkin",
    "working": "workin",
    "making": "makin",
    "taking": "takin",
    "finding": "findin",
    "thinking": "thinkin",
    "feeling": "feelin",
    
    # Additional informal contractions
    "want a": "wanna",
    "got a": "gotta",
    "lot of": "lotta",
    "trying to": "tryna",
    "going to be": "gonna be",
    "supposed to": "s'posed to",
    "about to": "bout to",
}


# Pre-compile regex patterns for speed (case-insensitive)
PATTERNS = [
    (re.compile(r'\b' + re.escape(expansion) + r'\b', flags=re.IGNORECASE), contraction)
    for expansion, contraction in CONTRACTION_MAP.items()
]

def sms_chat_contraction_noise(sentence: str) -> str:
    """
    Replace common two-token phrases with their SMS/chat contraction equivalents.
    E.g. "I am still waiting" → "I'm still waiting"
    """
    noisy = sentence
    for pattern, contraction in PATTERNS:
        noisy = pattern.sub(contraction, noisy)
    return noisy




# Only Cyrillic look-alikes (Latin → Cyrillic)
CYRILLIC_LOOKALIKES = {
    'A': 'А',  'a': 'а',   # A
    'B': 'В',  'b': 'ь',   # B → Ve, b → soft sign
    'C': 'С',  'c': 'с',   # C
    'E': 'Е',  'e': 'е',   # E
    'H': 'Н',  'h': 'һ',   # H → En, h → Shha
    'I': 'І',  'i': 'і',   # I
    'J': 'Ј',  'j': 'ј',   # J
    'K': 'К',  'k': 'к',   # K
    'M': 'М',  'm': 'м',   # M
    'O': 'О',  'o': 'о',   # O
    'P': 'Р',  'p': 'р',   # P
    'S': 'Ѕ',  's': 'ѕ',   # S
    'T': 'Т',  't': 'т',   # T
    'U': 'У',  'u': 'у',   # U
    'V': 'Ѵ',  'v': 'ѵ',   # V → Izhitsa
    'X': 'Х',  'x': 'х',   # X
    'Y': 'Ү',  'y': 'ү',   # Y → Straight U
}

_word_punct = re.compile(r'^([A-Za-z]+)(\W*)$')  # only ASCII letters

def cyrillic_homoglyph_noise(sentence: str) -> str:
    tokens   = sentence.split(' ')
    candidates = []
    for i, tok in enumerate(tokens):
        m = _word_punct.match(tok)
        if not m:
            continue
        word, punct = m.group(1), m.group(2)
        # now word is guaranteed ASCII only
        if any(ch in CYRILLIC_LOOKALIKES for ch in word):
            candidates.append((i, word, punct))

    if not candidates:
        return sentence

    idx, word, punct = random.choice(candidates)
    noised = ''.join(
        CYRILLIC_LOOKALIKES.get(ch, ch)
        for ch in word
    )
    tokens[idx] = noised + punct
    return ' '.join(tokens)

def remove_cues_noise(sentence: str) -> str:
    """
    Lower-case the sentence and strip out all punctuation cues
    (e.g. apostrophes, question marks, commas, etc.), 
    collapsing multiple spaces into one.
    
    E.g.:
      "What can I do if my card still hasn’t arrived after 2 weeks?"
        → "what can i do if my card still hasnt arrived after 2 weeks"
    """
    # 1) Normalize to lower case
    s = sentence.lower()
    # 2) Remove any character that's not a-z, 0-9, or whitespace
    s = re.sub(r'[^a-z0-9\s]', '', s)
    # 3) Collapse any sequence of whitespace (including non-breaking spaces) into a single space
    s = re.sub(r'\s+', ' ', s)
    # 4) Strip leading/trailing spaces
    return s.strip()

nlp = spacy.load("en_core_web_sm")

def pronoun_swap_noise(text: str) -> str:
    """
    Replace any occurrence of "my <noun>" or "the <noun>" in the text with "it" (preserving case),
    based on the first noun chunk found in the entire text.

    Examples:
      "I am still waiting on my card?"         → "I am still waiting on it?"
      "Is the card still coming?"               → "Is it still coming?"
      "My new card hasn't arrived."             → "It hasn't arrived."
      "Does the package with my card ..."       → "Does the package with it ..."
    """
    # Analyze whole text for noun chunks
    doc = nlp(text)
    target = None
    for chunk in doc.noun_chunks:
        lc = chunk.text.lower()
        if lc.startswith("my ") or lc.startswith("the "):
            target = chunk.text
            break
    if not target:
        return text

    # Prepare regex to match the exact phrase
    pattern = re.compile(r"\b" + re.escape(target) + r"\b", flags=re.IGNORECASE)

    def _replace(match):
        tok = match.group(0)
        # preserve capitalization
        return "It" if tok[0].isupper() else "it"

    # Substitute across entire text
    return pattern.sub(_replace, text)


def add_random_space(sentence):
    """
    A simpler alternative that preserves all existing spaces and adds one more
    at a random position between words.
    
    Args:
        sentence (str): The input sentence
        
    Returns:
        str: The sentence with a random extra space added
    """
    # Find all positions where there's a space
    space_positions = [i for i, char in enumerate(sentence) if char == ' ']
    
    # If no spaces found, return the original
    if not space_positions:
        return sentence
    
    # Choose a random space position
    random_pos = random.choice(space_positions)
    
    # Insert an additional space at that position
    result = sentence[:random_pos] + " " + sentence[random_pos:]
    
    return result

def remove_random_space(sentence):
    """
    Removes one space at a random position from the sentence.
    
    Args:
        sentence (str): The input sentence
        
    Returns:
        str: The sentence with a random space removed, or the original if no spaces
    """
    # Find all positions where there's a space
    space_positions = [i for i, char in enumerate(sentence) if char == ' ']
    
    # If no spaces found, return the original
    if not space_positions:
        return sentence
    
    # Choose a random space position
    random_pos = random.choice(space_positions)
    
    # Remove a space at that position by concatenating the parts before and after
    result = sentence[:random_pos] + sentence[random_pos+1:]
    
    return result
# end noise defs



# Character-level noise helpers
extra_words = ["very", "really", "now", "indeed", "just", "perhaps", "actually"]
phonetic_map = {
    'k':'q','q':'k','i':'y','y':'i','e':'a','a':'e','o':'u','u':'o',
    's':'z','z':'s','t':'d','d':'t','b':'v','v':'b','p':'b','c':'k',
    'm':'n','n':'l','l':'m','j':'g','g':'j','f':'v','h':'a'
}
cyrillic_map = {
    'a':'а','b':'в','c':'с','d':'д','e':'е','f':'ғ','g':'ɡ','h':'н',
    'i':'і','j':'ј','k':'к','l':'ⅼ','m':'м','n':'п','o':'о','p':'р',
    'q':'զ','r':'г','s':'ѕ','t':'т','u':'υ','v':'ѵ','w':'ш','x':'х','y':'у','z':'ᴢ'
}

def delete_word(text):
    w=text.split(); return " ".join(w[:(i:=random.randrange(len(w)))] + w[i+1:]) if len(w)>1 else text


def swap_neighbors(text):
    w=text.split();
    if len(w)>=2:
        i=random.randrange(len(w)-1); w[i],w[i+1]=w[i+1],w[i]
        return " ".join(w)
    return text


def semantic_replace(text):
    try: return EmbeddingAugmenter().augment(text)[0]
    except: return text


def double_word(text):
    w=text.split(); i=random.randrange(len(w)) if w else 0; w.insert(i,w[i])
    return " ".join(w)


def insert_word(text):
    w=text.split(); i=random.randrange(len(w)+1); w.insert(i,random.choice(extra_words))
    return " ".join(w)


def delete_char(text):
    if len(text)>1: i=random.randrange(len(text)); return text[:i]+text[i+1:]
    return text


def swap_adj_chars(text):
    if len(text)>=2: i=random.randrange(len(text)-1); c=list(text); c[i],c[i+1]=c[i+1],c[i]; return "".join(c)
    return text


def double_char(text):
    if text: i=random.randrange(len(text)); c=list(text); c.insert(i,c[i]); return "".join(c)
    return text


def phonetic_char(text):
    if text:
        i=random.randrange(len(text)); ch=text[i].lower()
        if ch in phonetic_map:
            r=phonetic_map[ch]; r=r.upper() if text[i].isupper() else r
            return text[:i]+r+text[i+1:]
    return text

def keyboard_char(text):
    if text:
        i=random.randrange(len(text))
        text = keyboard_noise_i(text, i)
    return text


def insert_char(text): i=random.randrange(len(text)+1); return text[:i]+chr(random.randint(97,122))+text[i:]

def cyrillic_char(text):
    if text:
        i=random.randrange(len(text)); ch=text[i].lower()
        if ch in cyrillic_map:
            r=cyrillic_map[ch]; r=r.upper() if text[i].isupper() else r
            return text[:i]+r+text[i+1:]
    return text


def remove_punctuation(text): return text.translate(str.maketrans('','','') if False else str.maketrans('','',string.punctuation))

def random_capitalization(text):
    letters=[i for i,c in enumerate(text) if c.isalpha()]
    if letters:
        i=random.choice(letters); c=text[i]; return text[:i]+(c.upper() if c.islower() else c.lower())+text[i+1:]
    return text


def normalize_spaces(text):
    if random.random()<.5:
        sp=[i for i,c in enumerate(text) if c==' ']
        if sp:
            i=random.choice(sp); return text[:i]+text[i+1:]
    i=random.randrange(len(text)+1); return text[:i]+' '+text[i:]


def drop_after_comma(text): return text.split(',')[0] if ',' in text else text


digit_map={**{str(i):w for i,w in enumerate(["zero","one","two","three","four",
                                              "five","six","seven","eight","nine"])},
           **{w:str(i) for i,w in enumerate(["zero","one","two","three","four",
                                              "five","six","seven","eight","nine"])}}

def replace_digits_with_words(text):
    changed=False; out=[]
    for ch in text:
        if ch.isdigit(): out.append(digit_map[ch]); changed=True
        else: out.append(ch)
    if changed: return "".join(out)
    w=text.split()
    for j,tk in enumerate(w):
        core=tk.strip(string.punctuation).lower()
        if core in digit_map and digit_map[core].isdigit():
            w[j]=tk.replace(core,digit_map[core])
    return " ".join(w)

#-------------noise char i defs -----------------##
def delete_char_i(s, i):
    """
    Delete the character at index i in string s.
    If i is out of bounds, return s unchanged.
    """
    if 0 <= i < len(s):
        return s[:i] + s[i+1:]
    return s


def swap_char_i(s, i):
    """
    Swap the character at index i with the character at i+1 in string s.
    If i is the last index or out of bounds, return s unchanged.
    """
    if 0 <= i < len(s) - 1:
        chars = list(s)
        chars[i], chars[i+1] = chars[i+1], chars[i]
        return ''.join(chars)
    return s

def double_char_i(s, i):
    """
    Double (repeat) the character at index i in string s.
    If i is out of bounds, return s unchanged.
    """
    if 0 <= i < len(s):
        return s[:i] + s[i] + s[i] + s[i+1:]
    return s

def phonetic_char_i(text, i):
    if text and 0 <= i < len(text):
        ch=text[i].lower()
        if ch in phonetic_map:
            r=phonetic_map[ch]; r=r.upper() if text[i].isupper() else r
            return text[:i]+r+text[i+1:]
    return text

def insert_char_after_i(text, i): 
    if 0 <= i < len(text):
        return text[:i+1]+chr(random.randint(97,122))+text[i+1:]
    return text

def cyrillic_char_i(text, i):
    if text and 0 <= i < len(text):
        ch=text[i].lower()
        if ch in cyrillic_map:
            r=cyrillic_map[ch]; r=r.upper() if text[i].isupper() else r
            return text[:i]+r+text[i+1:]
    return text


def random_capitalization_i(text, i):
    if text and 0 <= i < len(text):
        c=text[i]; return text[:i]+(c.upper() if c.islower() else c.lower())+text[i+1:]
    return text

keyboard_neighbors = {
    'a': ['q','w','s','z'],    'b': ['v','g','h','n'],
    'c': ['x','d','f','v'],    'd': ['s','e','r','f','c','x'],
    'e': ['w','s','d','r'],    'f': ['d','r','t','g','v','c'],
    'g': ['f','t','y','h','b','v'],'h': ['g','y','u','j','n','b'],
    'i': ['u','j','k','o'],     'j': ['h','u','i','k','m','n'],
    'k': ['j','i','o','l','m'], 'l': ['k','o','p'],
    'm': ['n','j','k'],         'n': ['b','h','j','m'],
    'o': ['i','k','l','p'],     'p': ['o','l'],
    'q': ['w','a'],             'r': ['e','d','f','t'],
    's': ['a','w','e','d','x','z'],'t': ['r','f','g','y'],
    'u': ['y','h','j','i'],     'v': ['c','f','g','b'],
    'w': ['q','a','s','e'],     'x': ['z','s','d','c'],
    'y': ['t','g','h','u'],     'z': ['a','s','x']
}

def keyboard_noise_i(s, i):
    """
    Replace s[i] with a random neighboring key on a QWERTY layout.
    If s[i] is not an alphabetic character or has no neighbors, or i is
    out of bounds, returns s unchanged.
    """
    if 0 <= i < len(s):
        ch = s[i]
        # only apply to alphabetic letters
        if ch.isalpha():
            neighs = keyboard_neighbors.get(ch.lower())
            if neighs:
                sub = random.choice(neighs)
                # preserve original casing
                sub = sub.upper() if ch.isupper() else sub
                return s[:i] + sub + s[i+1:]
    return s
                
