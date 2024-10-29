from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose the model you want
model_name = "facebook/opt-2.7b"  # or any of the other options

local_dir = "./opt-2.7B/models--facebook--opt-2.7b/snapshots/905a4b602cda5c501f1b3a2650a4152680238254"
# Load the tokenizer and model from hugging face
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="opt-2.7B")
# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="opt-2.7B")

# load the model from local dir
tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(local_dir)

input_text = "The possibilities of artificial intelligence are endless,"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text
output = model.generate(**inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
