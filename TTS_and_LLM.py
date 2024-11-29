import asyncio
import torch
import os
from llama_cpp import Llama
import pyttsx3

# LLM and TTS model paths
model_path = "mistral-7b-openorca.Q3_K_M.gguf"

# Load the LLM model and tokenizer asynchronously
async def load_llm_model():
    print("Loading LLM model...")
    llm = Llama(model_path=model_path, n_gpu_layers=1, n_ctx=4096)
    print("LLM model loaded.")
    return llm

# Async function to generate text using LLM and speak as we generate it
async def generate_and_speak(tts, llm, input_text, max_tokens=500):
    output = llm.create_completion(f"""<|im_start|>system
    You are a helpful chatbot.
    <|im_end|>
    <|im_start|>user
    {input_text}<|im_end|>
    <|im_start|>""", max_tokens=max_tokens, stop=["<|im_end|>"], stream=True)

    generated_text = ""
    for token in output:
        text_chunk = token["choices"][0]["text"]
        generated_text += text_chunk
        print(text_chunk, end='', flush=True)
        
        # Speak the generated part of the text
        await speak_text(tts, text_chunk)
    
    return generated_text

# Async function to load TTS engine
async def load_TTS():
    tts = pyttsx3.init()
    
    # Set speech rate to a more moderate speed (default rate is often too fast)
    rate = tts.getProperty('rate')
    tts.setProperty('rate', rate - 50)  # Lower the rate (you can adjust this value as needed)
    
    return tts

# Async function to perform TTS and play audio
async def speak_text(tts_model, text_to_speak):
    tts_model.say(text_to_speak)
    tts_model.runAndWait()

# Main async function to coordinate everything
async def main(input_text):
    # Load models
    llm = await load_llm_model()
    tts = await load_TTS()

    # Generate and speak text from LLM as it's being generated
    await generate_and_speak(tts, llm, input_text)

# Run the async main function
input_text = "write me a poem about AI"  # Example input
asyncio.run(main(input_text))
