from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Step 1: Load the GPT-2 Large model and tokenizer
model_name = "gpt2-large"
local_dir = "./large-gpt-2/models--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519"

print("Loading the model ...")
tokenizer = GPT2Tokenizer.from_pretrained(local_dir)
model = GPT2LMHeadModel.from_pretrained(local_dir)

# Step 2: Generate text with the model
input_text = "Tell me a bit about who you're and what you're capable of"
print("received the input")
inputs = tokenizer(input_text, return_tensors="pt")
print("tokenizing the input")

# Generate text with a maximum length of 50 tokens
output = model.generate(**inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
print("generating output .. ")
# Step 3: Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
