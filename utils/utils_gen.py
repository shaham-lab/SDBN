"""
utils_gen.py — Shared utilities for the generative (decoder-only) pipeline.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, set_peft_model_state_dict


def load_lora_gen_model(model_name, save_path, lora_rank=4, device='cuda:0'):
    """Load a saved LoRA decoder-only model (LLaMA/Qwen) for inference."""
    print(f"Loading model from: {save_path}")
    state_dict = torch.load(save_path, map_location='cpu')

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
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
