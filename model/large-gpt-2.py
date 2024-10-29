from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = "gpt2-large"

tokenizer = GPT2Tokenizer.from_pretrained(model, cache_dir="./large-gpt2")
model = GPT2LMHeadModel.from_pretrained(model, cache_dir="./large-gpt2")

input_text = "The possibilities of artificial intelligence are endless,"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text with a maximum length of 50 tokens
output = model.generate(**inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

# Step 3: Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)