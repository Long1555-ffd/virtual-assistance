from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1")

inputs = tokenizer("How can I help you today?", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
