from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Step 1: Load the GPT-2 Medium model and tokenizer
model_name = "gpt2-medium"
local_dir = "./gpt-2-medium/models--gpt2-medium/snapshots/6dcaa7a952f72f9298047fd5137cd6e4f05f41da"
tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir='./gpt-2-medium')
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir='./gpt-2-medium')

# Step 2: Generate text with the model
input_text = "Tell me about the future of AI and robotics, can AI be the big brain for robots to process their surrounding?"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text with a maximum length of 50 tokens
output = model.generate(**inputs, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)

# Step 3: Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
