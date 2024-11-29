from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

# Step 1: Load the GPT-2 Medium model and tokenizer
model_name = "gpt2-medium"
local_dir = "./gpt-2-medium/models--gpt2-medium/snapshots/6dcaa7a952f72f9298047fd5137cd6e4f05f41da"

if os.path.isdir(local_dir):
    tokenizer = GPT2Tokenizer.from_pretrained(local_dir)
    model = GPT2LMHeadModel.from_pretrained(local_dir)
else:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir='./gpt-2-medium')
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir='./gpt-2-medium')

# Step 2: Generate text with the model
input_text = "Tell me all the steps to make a virtual assistance for myself"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text with a maximum length of 50 tokens
output = model.generate(**inputs, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2)

# Step 3: Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
