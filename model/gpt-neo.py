from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model = "EleutherAI/gpt-neo-1.3B"
local_dir = "./gpt-neo-1.3B/models--EleutherAI--gpt-neo-1.3B"

tokenizer = GPT2Tokenizer.from_pretrained(model, cache_dir="gpt-neo-1.3B")
model = GPTNeoForCausalLM.from_pretrained(model, cache_dir="gpt-neo-1.3B")

input_text = "The possibilities of artificial intelligence are endless,"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text with a maximum length of 50 tokens
output = model.generate(**inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

# Step 3: Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)