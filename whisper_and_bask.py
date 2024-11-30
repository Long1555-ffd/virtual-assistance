import os
import whisper
from bark import generate_audio, preload_models
from transformers import pipeline
import torch
from scipy.io.wavfile import write

# from llama_cpp import Llama
# Define devices
device1 = "cuda" if torch.cuda.is_available() else "cpu"
device2 = "cpu"  # Bark runs on CPU by default

# llm = Llama(model_path=model_path, n_gpu_layers=1, n_ctx=4096)

# def generate_response(llm, input_text, max_tokens=500):
#     output = llm.create_completion(f"""<|im_start|>system
#     You are a helpful chatbot.
#     <|im_end|>
#     <|im_start|>user
#     {input_text}<|im_end|>
#     <|im_start|>""", max_tokens=max_tokens, stop=["<|im_end|>"], stream=True)
#     generated_text = ""
#     for token in output:
#         text_chunk = token["choices"][0]["text"]
#         generated_text += text_chunk
#         print(text_chunk, end='', flush=True)
#     return generated_text

whisper_model = whisper.load_model("small", device=device1, download_root="./whisper")  # Use "large" for better accuracy
print(f"Model loaded on the {device1}")

# Preload Bark Models
preload_models()  # Bark models are preloaded and cached locally

# Use a lightweight LLM for reasoning
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=0 if device == "cuda" else -1)


# Function: Transcribe Speech (Whisper)
def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result["text"]

# Function: Generate Response (LLM or Logic)
def generate_response(text):
    # Modify this logic to integrate custom logic or your LLM.
    response = qa_pipeline(text)[0]["generated_text"]
    return response

# Function: Convert Text to Speech (Bark)
def text_to_speech(text, output_path):
    audio_array = generate_audio(text)
    unnormalized_data = audio_array*32767
    audio_array = np.int16(unnormalized_data)
    print(audio_array)
    # with open(output_path, "wb") as f:
    #     audio_array.tofile(f)
    write(output_path, 22050, audio_array)

# Main Function
def run_pipeline(input_audio_path, output_audio_path):
    # Step 1: Transcribe speech
    print("Transcribing audio...")
    transcribed_text = transcribe_audio(input_audio_path)
    print(f"Transcribed Text: {transcribed_text}")

    # Step 2: Generate response
    print("Generating response...")
    response_text = generate_response(transcribed_text)
    print(f"Response Text: {response_text}")

    # Step 3: Convert response to speech
    print("Generating speech...")
    text_to_speech(response_text, output_audio_path)
    print(f"Audio output saved to: {output_audio_path}")

# Example Usage
if __name__ == "__main__":
    input_audio = "example-TTS.wav"  # Path to input audio file
    output_audio = "output.mp3"  # Path to save output audio
    run_pipeline(input_audio, output_audio)