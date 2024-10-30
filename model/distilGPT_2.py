from transformers import AutoModelForCausalLM, AutoTokenizer

# Load distilGPT-2 model and tokenizer
model_name = "distilgpt2"

# Download the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Test the model with a simple input
input_text = "Who are you"
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs, max_length=50, num_return_sequences=1)

# Decode and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
