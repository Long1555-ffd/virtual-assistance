from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import os
model = "EleutherAI/gpt-neo-1.3B"
local_dir = "./gpt-neo-1.3B/models--EleutherAI--gpt-neo-1.3B/snapshots/dbe59a7f4a88d01d1ba9798d78dbe3fe038792c8"
device = "auto"
try:
    print("Trying to load the model from local directory")
    tokenizer = GPT2Tokenizer.from_pretrained(local_dir)
    model = GPTNeoForCausalLM.from_pretrained(local_dir).to(device)
except Exception as e:
    print("model is not available in your working folder, downloading the model from huggingface")
#     tokenizer = GPT2Tokenizer.from_pretrained(model, cache_dir="gpt-neo-1.3B")
#     model = GPTNeoForCausalLM.from_pretrained(model, cache_dir="gpt-neo-1.3B").to(device)

input_text = "Tell me about the possibilities of AI application in education, will it replace human teachers?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate text with a maximum length of 50 tokens
output = model.generate(**inputs, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)

# Step 3: Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)