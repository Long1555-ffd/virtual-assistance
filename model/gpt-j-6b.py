from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose the model you want
model_name = "EleutherAI/gpt-j-6B"  # or the LLaMA 2 or OPT-1.3B if you have access

device = "cuda"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./gpt-j-6b").to(device)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./gpt-j-6b")

input_text = "The possibilities of artificial intelligence are endless,"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text
output = model.generate(**inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
