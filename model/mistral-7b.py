import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

device = "cuda"

model = AutoModelForCausalLM.from_pretrained("Open-Orca/Mistral-7B-OpenOrca").to(device)
tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")