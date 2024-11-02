import asyncio
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import sounddevice as sd

# LLM and TTS model paths
llm_model_name = "gpt2-medium"
tts_model_name_or_path = "innoai/Edge-Text-To-Speech"
tts_local_dir = "./edge_text_to_speech"

# Load the LLM model and tokenizer asynchronously
async def load_llm_model():
    print("Loading LLM model...")
    tokenizer = GPT2Tokenizer.from_pretrained(llm_model_name)
    model = GPT2LMHeadModel.from_pretrained(llm_model_name)
    print("LLM model loaded.")
    return tokenizer, model

# Async function to generate text using LLM
async def generate_text(llm_tokenizer, llm_model, input_text):
    print("Generating text...")
    inputs = llm_tokenizer(input_text, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Text generation complete.")
    return generated_text

# Load TTS model asynchronously
async def load_tts_model():
    print("Loading TTS model...")
    processor = AutoProcessor.from_pretrained(tts_model_name_or_path, cache_dir=tts_local_dir)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(tts_model_name_or_path, cache_dir=tts_local_dir)
    print("TTS model loaded.")
    return processor, model

# Async function to perform TTS and play audio
async def speak_text(tts_processor, tts_model, text):
    print("Converting text to speech...")
    inputs = tts_processor(text=text, return_tensors="pt")
    with torch.no_grad():
        speech = tts_model.generate(**inputs)
    print("Playing speech...")
    sample_rate = 22050  # Adjust as needed
    sd.play(speech.numpy(), samplerate=sample_rate)
    sd.wait()
    print("Speech playback completed.")

# Main async function to coordinate everything
async def main():
    # Load models
    llm_tokenizer, llm_model = await load_llm_model()
    tts_processor, tts_model = await load_tts_model()

    # Generate text from LLM
    input_text = "Tell me all the steps to make a virtual assistant for myself"
    generated_text = await generate_text(llm_tokenizer, llm_model, input_text)
    print("Generated text:", generated_text)

    # Convert generated text to speech
    await speak_text(tts_processor, tts_model, generated_text)

# Run the async main function
asyncio.run(main())
