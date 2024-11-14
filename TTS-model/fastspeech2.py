from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import torch
import IPython.display as ipd

# Load the model and task
models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)
model = models[0]
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)

# Build the generator
generator = task.build_generator(model, cfg)

# Text for TTS
text = "Hello, this is a test run."

# Prepare the model input
sample = TTSHubInterface.get_model_input(task, text)

# Ensure the model is in evaluation mode
model.eval()

# Get the output waveform
with torch.no_grad():  # Disable gradient computation for inference
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

# Play the output audio
ipd.Audio(wav, rate=rate)

# from transformers import pipeline

# # Specify the model name from Hugging Face's model hub
# model_name = "facebook/fastspeech2-en-ljspeech"

# # Download the model and save it locally
# tts = pipeline("text-to-speech", model=model_name, cache_dir="./tts_models")
# print("Model downloaded and saved in ./tts_models folder.")

# # Example text
# text = "Hello, this is a test of text-to-speech conversion."

# # Convert text to speech
# audio_output = tts(text)

# # Save the output audio to a file
# with open("output.wav", "wb") as f:
#     f.write(audio_output["audio"])
# print("Audio saved as output.wav")
