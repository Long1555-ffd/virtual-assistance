import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

device = "cuda"

model = AutoModelForCausalLM.from_pretrained("Open-Orca/Mistral-7B-OpenOrca").to(device)
tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")

inputs = tokenizer("Who are you?", return_tensors="pt").to(device)
outputs = model.generate(
    **inputs, max_new_tokens=256, use_cache=True, do_sample=True,
    temperature=0.2, top_p=0.95
)
text = tokenizer.batch_decode(outputs)[0]
print(text)
