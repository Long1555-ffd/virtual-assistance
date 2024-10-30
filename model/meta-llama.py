from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model name and local directory
model_name = "facebook/opt-2.7b"
local_dir = "./opt-2.7B/models--facebook--opt-2.7b/snapshots/905a4b602cda5c501f1b3a2650a4152680238254"

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

# Load the tokenizer and model from the local directory
try:
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForCausalLM.from_pretrained(local_dir)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model/tokenizer: {e}")
    exit()

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Input text
input_text = "Tell me about the future of AI and robotics, can AI be the big brain for robots to process their surroundings?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate text
try:
    output = model.generate(**inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    print("Text generation successful.")
except Exception as e:
    print(f"Error during text generation: {e}")
    exit()

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:")
print(generated_text)

