import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, hf_token: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=hf_token
    )
    return model


def initialize_tokenizer(model_name: str, hf_token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    tokenizer.bos_token_id = 1
    return tokenizer
