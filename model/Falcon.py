from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")

inputs = tokenizer("What can I assist you with today?", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
