import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import init_empty_weights, infer_auto_device_map
import warnings

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
model_name = "facebook/opt-2.7B"
cache_dir = "./optimized-opt-2.7B-8bit"
local_dir = "./optimized-opt-2.7B-8bit/models--facebook--opt-2.7B/snapshots/905a4b602cda5c501f1b3a2650a4152680238254"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# Initialize model with empty weights to manage device map efficiently
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    # tie the model weights to prevent memory overhead
    model.tie_weights()

# Infer device map to manage model layers between GPU and CPU
device_map = infer_auto_device_map(model, max_memory={0: "4GiB", "cpu": "6GiB"})

# configure quantization using BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Load the model with 8-bit quantization and mixed precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    device_map="auto",
    quantization_config=quantization_config,
    offload_folder="./offload",
    torch_dtype=torch.float16,
    offload_state_dict=True
)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

def generate_text(prompt, max_length=500):
    # Move inputs to the appropriate device (CPU or GPU)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate text using the optimized model
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

    # Decode the outputs and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)


