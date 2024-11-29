import os
import sys
from contextlib import redirect_stdout
from llama_cpp import Llama

os.environ["LLAMA_LOG_LEVEL"] = "ERROR"

# Suppress stdout during model loading
with open(os.devnull, "w") as f, redirect_stdout(f):
    llm = Llama(model_path="mistral-7b-openorca.Q3_K_M.gguf",
                n_gpu_layers=1, n_ctx=4096)

# Generate output
output = llm.create_completion("""<|im_start|>system
You are a helpful chatbot.
<|im_end|>
<|im_start|>user
tell me a poem about AGI<|im_end|>
<|im_start|>assistant""", max_tokens=500, stop=["<|im_end|>"], stream=True)

# Print the result
for token in output:
    print(token["choices"][0]["text"], end='', flush=True)
