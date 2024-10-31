import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.exporters.onnx import export_models
import onnxruntime as ort

model_name = "Open-Orca/Mistral-7B-OpenOrca"
cache_dir = "./model-mistral-cache"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)

# Export the llama 2 model to onnx format using optimum
onnx_path = "./mistral-7b.onnx"

# load the model in the pytorch format
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

# export the model into the onnx format
onnx_dir = "./onnx_models"
os.makedirs(onnx_dir, exist_ok=True)  # Create directory if it doesn't exist
export_models(
    model=model,
    output_dir=onnx_dir,  # Change this to a directory
    opset=12,
    cache_dir=cache_dir
)

# apply the dynamtic quantization using optimum
quant_config = AutoQuantizationConfig.avx512_vnni(is_static=False)  

# quantize the ONNX model
quantized_model_path = "./quantized-mistral-7B.onnx"
quantized_model = ORTModelForCausalLM.from_pretrained(
    onnx_path,
    export=True,
    save_directory=quantized_model_path,
    quantization_config = quant_config,
    cache_sir=cache_dir
)

# set up ONNX runtime with GPU and CPU offloading
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# initialize an ONNX Runtime session with the quantized model
session_options = ort.SessionOptions()
ort_session = ort.InferenceSession(quantized_model_path, providers=providers, sess_options=session_options)

def generate_text(prompt, max_length=100):
    print(f"Generating text for prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.numpy()  # convert to numpy for onnx compatibility

    onnx_inputs = {"input_ids": input_ids}

    # generate text with ONNX runtime
    outputs = ort_session.run(None, onnx_inputs)
    generated_ids = outputs[0]

    print(f"Generated IDs: {generated_ids}")
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

prompt = "The future of AI research is"
try: 
    generate_text(prompt)
except Exception as e:
    print(e)
    