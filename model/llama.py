from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

inputs = tokenizer("How can I assist you today?", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))