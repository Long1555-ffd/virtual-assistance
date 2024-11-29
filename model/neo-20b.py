from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose the model you want
model_name = "EleutherAI/gpt-neox-20b"  # or any of the other options

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./gpt-neo-20b")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./gpt-neo-20b")

input_text = "The possibilities of artificial intelligence are endless,"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text
output = model.generate(**inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
