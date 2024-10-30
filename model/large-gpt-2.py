from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = "gpt2-large"
local_dir = "./large-gpt-2/models--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519"
tokenizer = GPT2Tokenizer.from_pretrained(local_dir)
model = GPT2LMHeadModel.from_pretrained(local_dir)

input_text = "Tell me about the future of AI and robotics, can AI be the big brain for robots to process their surrounding?"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text with a maximum length of 50 tokens
output = model.generate(**inputs, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)

# Step 3: Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)