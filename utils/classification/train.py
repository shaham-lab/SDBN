#train.py classification
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, LoraModel
from datasets import load_dataset, DatasetDict
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
import utils
import numpy as np

device = None
models_dict = {
    'bert-b' : 'bert-base-uncased',
    'llama' : 'meta-llama/Llama-2-7b-hf',
    'mbert' : 'bert-base-multilingual-uncased',
    'deberta' : 'microsoft/deberta-v3-base',
    'deberta-l' : "microsoft/deberta-v3-large",
    'mdeberta' : 'microsoft/mdeberta-v3-base'
}

def collate_fn(batch):
    # batch is a list of ((premise, hyp), label)
    pairs, labels = zip(*batch)
    # pairs is a tuple of (premise, hyp) tuples → convert to list
    return list(pairs), torch.tensor(labels, dtype=torch.long)

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='banking77', help='dataset for training: banking77, trec, bless, news, imdb, nli, ArSarcasm-v2')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=10, help='number of adversarial training epochs')
    parser.add_argument('--warm_up', type=int, default=3, help='number of clean warm-up epochs')
    parser.add_argument('--alg', type=str, default='lora', help='PEFT method: lora, qlora, bitfit, adapter, full_ft')
    parser.add_argument('--model', type=str, default='bert-b', help='pretrained model key')
    parser.add_argument('--rank', type=int, default=12, help='LoRA rank (12 for bert, 4 for deberta)')
    parser.add_argument('--output', '-o', type=str, default='')
    parser.add_argument('--device', type=str, default='0', help='CUDA device index')
    parser.add_argument('--init_seed', type=int, default=0, help='random seed')
    parser.add_argument('--train_set_size', default=1.0, type=float, help='fraction of training data to use')
    parser.add_argument('--epsilon', '-ep', type=float, default=1e-4, help='adversarial perturbation magnitude')
    parser.add_argument('--task', type=str, default='', help='sub-task (e.g. for ArSarcasm: sentiment_egypt, sentiment_levant)')
    parser.add_argument('--pertubation', '-pt', default='sdbn', type=str, choices=['sdbn', 'sdbn-h'],
                        help='adversarial perturbation type: sdbn (embedding l-inf FGSM) or sdbn-h (hybrid with character-level)')

    args = parser.parse_args()
    print("Args:", args)

    mode = 'adv'

    if args.rank == 12:
        rank = ''
    else:
        rank = f"rank_{args.rank}"

    alg_dir = f"{args.alg}_{args.pertubation}"
    output_dir = os.path.join('.', 'results', 'SDBN', f'Ep_{args.epochs}',
                              args.dataset + args.task, f"percent_{args.train_set_size}",
                              args.model, alg_dir, rank, args.output)
    print("output_dir:", output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    utils.set_random_seed(args.init_seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    utils.device = device
    print(f"Using device: {device}")

    # Create tokenizer
    if 'deberta' in args.model:
        tokenizer = AutoTokenizer.from_pretrained(models_dict[args.model])
    else:
        tokenizer = BertTokenizer.from_pretrained(models_dict[args.model])

    accuracy_train_list = []
    accuracy_test_list = []
    loss_train_list = []
    loss_test_list = []

    print("\nLoading dataset:", args.dataset)
    if args.dataset == 'banking77':
        # Use data_files parameter to bypass the script issue
        #dataset = load_dataset("csv", data_files={"train":"banking77/train.csv", "test":"banking77/test.csv"}) #load_dataset("PolyAI/banking77")
        # Use data_files parameter to bypass the script issue
        raw_dataset = load_dataset("csv", data_files={"train":"banking77/train.csv", "test":"banking77/test.csv"})
            
        # Convert 'category' field to 'label' using the label map
        dataset = DatasetDict({
            'train': {'text': raw_dataset['train']['text'], 'label': [utils.BANKING77_LABEL_MAP[cat] for cat in raw_dataset['train']['category']]},
            'test': {'text': raw_dataset['test']['text'], 'label': [utils.BANKING77_LABEL_MAP[cat] for cat in raw_dataset['test']['category']]}
        })
        
    elif args.dataset == 'nli':
        from datasets import Dataset as HFDataset, DatasetDict as HFDatasetDict
        raw = load_dataset("nyu-mll/multi_nli")
        fiction_train = raw["train"].filter(lambda ex: ex["genre"] == "fiction")
        fiction_val = raw["validation_matched"].filter(lambda ex: ex["genre"] == "fiction")
        train_ds = HFDataset.from_dict({
            "text": list(zip(fiction_train["premise"], fiction_train["hypothesis"])),
            "label": fiction_train["label"]
        })
        val_ds = HFDataset.from_dict({
            "text": list(zip(fiction_val["premise"], fiction_val["hypothesis"])),
            "label": fiction_val["label"]
        })
        dataset = HFDatasetDict({"train": train_ds, "test": val_ds})

    elif args.dataset == "trec":
        def load_trec_from_label_file(filepath):
            texts, labels = [], []
            with open(filepath, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            labels.append(parts[0])
                            texts.append(parts[1])
            return texts, labels
        train_texts, train_labels = load_trec_from_label_file("trec/train.label")
        test_texts, test_labels = load_trec_from_label_file("trec/test.label")
        label_to_id = utils.TREC_LABEL_MAP
        dataset = DatasetDict({
            'train': {'text': train_texts, 'label': [label_to_id[l] for l in train_labels]},
            'test': {'text': test_texts, 'label': [label_to_id[l] for l in test_labels]}
        })

    elif args.dataset == "imdb":
        dataset = load_dataset("imdb")
    elif args.dataset == "news":
        dataset = load_dataset("SetFit/20_newsgroups")
    elif args.dataset == 'ArSarcasm' or args.dataset == 'ArSarcasm-v2':
        if args.task == 'sentiment_egypt':
            train_ds, test_ds = utils.get_ArSarcasm_ds(
                drop_msa=True, ignore_list_dialects=['levant', 'magreb', 'gulf'])
        elif args.task == 'sentiment_levant':
            train_ds, test_ds = utils.get_ArSarcasm_ds(
                drop_msa=True, ignore_list_dialects=['egypt', 'magreb', 'gulf'])
        else:
            train_ds, test_ds = utils.get_ArSarcasm_ds(drop_msa=True)
        dataset = DatasetDict({
            'train': {'text': train_ds['tweet'], 'label': train_ds['sentiment']},
            'test': {'text': test_ds['tweet'], 'label': test_ds['sentiment']}
        })
    
    elif args.dataset == 'bless':
        raw_dataset = load_dataset("json", data_files={
            "train": "bless/bless_train.jsonl",
            "validation": "bless/bless_val.jsonl",
            "test": "bless/bless_test.jsonl",
        })
        train_label_set = set(raw_dataset['train']['relation'])
        label_to_id = {label: idx for idx, label in enumerate(sorted(train_label_set))}
        print(f"BLESS Label mapping: {label_to_id}")

        def format_text_with_sep(examples):
            return [f"{head} {tokenizer.sep_token} {tail}"
                    for head, tail in zip(examples['head'], examples['tail'])]

        train_texts = format_text_with_sep(raw_dataset['train'])
        train_labels_int = [label_to_id[l] for l in raw_dataset['train']['relation'] if l in label_to_id]
        test_texts_all = format_text_with_sep(raw_dataset['test'])
        test_texts, test_labels_int = [], []
        for i, label in enumerate(raw_dataset['test']['relation']):
            if label in label_to_id:
                test_texts.append(test_texts_all[i])
                test_labels_int.append(label_to_id[label])
        print(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")
        dataset = DatasetDict({
            'train': {'text': train_texts, 'label': train_labels_int},
            'test': {'text': test_texts, 'label': test_labels_int}
        })

    else:
        print(f"ERROR: invalid dataset '{args.dataset}'")
        exit(1)

    subset_df = pd.DataFrame(dataset['train']).sample(frac=args.train_set_size, random_state=args.init_seed)
    train_data = utils.SST5Dataset(pd.DataFrame(subset_df))

    test_data = utils.SST5Dataset(pd.DataFrame(dataset['test']))

    # Create data loaders
    if args.dataset == 'nli':
        train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=2,pin_memory=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_data,batch_size=args.batch_size,shuffle=False,num_workers=2,pin_memory=True, collate_fn=collate_fn)

    else:
        train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=2,pin_memory=True)
        test_loader = DataLoader(test_data,batch_size=args.batch_size,shuffle=False,num_workers=2,pin_memory=True)

    print(f"Dataset size: {len(train_data)} samples")
    print(f"Number of batches: {len(train_loader)}")
    
    print(f"Dataset size: {len(test_data)} samples")
    print(f"Number of batches: {len(test_loader)}")
    
    
    # Create model
    print("\nPreparing model")
    number_of_labels = len(set(dataset["train"]["label"]))
    print("number of classes:", number_of_labels)
    
    if args.model not in models_dict.keys():
        print(f"ERROR: {args.model} wasn't found in models_dict")
        exit(1)
    print("perpare model:", args.model)
    
    if args.model == 'llama':
        checkpointing_enable = True
    else:
        checkpointing_enable = False
        
    if args.alg == 'lora':
        model = utils.prepare_model_for_lora(number_of_labels, models_dict[args.model], args.rank)
    elif args.alg == 'qlora':
        model = utils.prepare_model_for_qlora(number_of_labels, models_dict[args.model], args.rank)
    elif args.alg == 'bitfit':
        model = utils.prepare_model_for_bitfit(number_of_labels, models_dict[args.model])
        args.lr = 1e-3
    elif args.alg == 'adapter':
        model = utils.prepare_model_for_adapter(number_of_labels, models_dict[args.model])
    elif args.alg == 'full_ft':
        model = utils.prepare_model_for_full_ft(number_of_labels, models_dict[args.model])
    else:
        print("ERROR: alg: {args.alg} wasn't found")
        exit(1)
        
    print("---Args:", args)
    
    model = model.to(device)
    print(model)
    
    # Count and display parameters
    total_params, trainable_params = utils.count_parameters(model)
    print("\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters (LoRA only): {trainable_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    print("\nStarting training...")
    print("Begin Warm Phase")
    for warm_epoch in range(args.warm_up):
        print("EPOCH: ", warm_epoch)
        train_loss, test_loss, train_acc, test_acc, model = utils.train_one_epoch(
            model, train_loader, test_loader, tokenizer, optimizer, device,
            checkpointing_enable=checkpointing_enable, alg=args.alg
        )
        
        print("\nEpoch completed!")
        print(f"Average train loss: {train_loss:.4f}")
        print(f"Average test loss: {test_loss:.4f}")
        print(f"Accuracy-Train: {train_acc:.2f}%")
        print(f"Accuracy-Test: {test_acc:.2f}%")
        loss_train_list.append(train_loss)
        loss_test_list.append(test_loss)
        accuracy_train_list.append(train_acc)
        accuracy_test_list.append(test_acc)
        
    print("Finish Warm Phase")
    
    print("Begin Adv Phase")

    for epoch in range(args.epochs):
        print("EPOCH: ", epoch + args.warm_up)
        adv_percent = 1.0
        train_loss, test_loss, train_acc, test_acc, model = utils.train_one_epoch(
            model, train_loader, test_loader, tokenizer, optimizer, device,
            adv_percent=adv_percent, alg=args.alg, checkpointing_enable=checkpointing_enable,
            epsilon=args.epsilon, pertubation=args.pertubation
        )
        print("\nEpoch completed!")
        print(f"Average train loss: {train_loss:.4f}")
        print(f"Average test loss: {test_loss:.4f}")
        print(f"Accuracy-Train: {train_acc:.2f}%")
        print(f"Accuracy-Test: {test_acc:.2f}%")
        loss_train_list.append(train_loss)
        loss_test_list.append(test_loss)
        accuracy_train_list.append(train_acc)
        accuracy_test_list.append(test_acc)

    print("Finish Adv Phase")        
   
        
    
    torch.save(loss_train_list, os.path.join(output_dir, f'loss_train_seed_{args.init_seed}_{mode}.pt'))
    torch.save(loss_test_list, os.path.join(output_dir, f'loss_test_seed_{args.init_seed}_{mode}.pt'))
    torch.save(accuracy_train_list, os.path.join(output_dir, f'acc_train_seed_{args.init_seed}_{mode}.pt'))
    torch.save(accuracy_test_list, os.path.join(output_dir, f'acc_test_seed_{args.init_seed}_{mode}.pt'))
    
    
    
    if args.alg == 'lora':
        utils.save_model_lora(model, output_dir, args.init_seed, args.dataset, mode)
    elif args.alg == 'qlora':
        utils.save_model_qlora(model, output_dir, args.init_seed, args.dataset, mode)
    elif args.alg == 'bitfit':
        utils.save_model_bitfit(model, output_dir, args.init_seed, args.dataset, mode)
    elif args.alg == 'adapter':
        utils.save_model_adapter(model, output_dir, args.init_seed, args.dataset, mode)
    elif args.alg == 'full_ft':
        utils.save_model_full_ft(model, output_dir, args.init_seed, args.dataset, mode)
    
    print("Fine Tune Done!")
    
   