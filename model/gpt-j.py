from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configure 8-bit quantization
quant_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir="./gpt-j-6b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", quantization_config=quant_config, device_map="cuda", cache_dir="./gpt-j-6b")


inputs = tokenizer("How can I assist you today?", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))