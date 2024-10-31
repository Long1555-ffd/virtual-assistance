import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.exporters.onnx import export_models
import onnxruntime as ort

model_name = "meta-llama/Llama-2-7b-hf"
cache_dir = "./model-cache"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# Export the llama 2 model to onnx format using optimum
onnx_path = "./llama-2.onnx"

# load the model in the pytorch format
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

# export the model into the onnx format
export_models(
    model=model,
    output_dir=onnx_path,
    opset=12,  # Onnx opset version (should be 12 or higher recommendation)
    task="causal-lm"
)

# apply the dynamtic quantization using optimum
quant_config = AutoQuantizationConfig.avx512_vnni(is_static=False)  

# quantize the ONNX model
quantized_model_path = "./quantized_llama2-7b.onnx"
quantized_model = ORTModelForCausalLM.from_pretrained(
    onnx_path,
    export=True,
    save_directory=quantized_model_path,
    quantization_config = quant_config
)

# set up ONNX runtime with GPU and CPU offloading
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# initialize an ONNX Runtime session with the quantized model
session_options = ort.SessionOptions()
ort_session = ort.InferenceSession(quantized_model_path, providers=providers, sess_options=session_options)

def generate_text(prompt, max_length = 100):
    inputs = tokenizer(prompt, return_tensor="pt")
    input_ids = inputs.input_ids.numpy()  # convert to numpy for onnx compatibility

    onnx_inputs = {"input_ids": input_ids}

    # generate text with ONNX runtime
    outputs = ort_session.run(None, onnx_inputs)
    generated_ids = outputs[0]

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

prompt = "The future of AI research is"
print(generate_text(prompt))