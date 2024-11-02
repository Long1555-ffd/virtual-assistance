from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch 
import sounddevice as sd

model_name = "innoai/Edge-TTS-Text-to-Speech"
local_dir = "./edge_text_to_speech" # the local dir where the model will be saved

try:
    print("loading the model from local dir")
    processor = AutoProcessor.from_pretrained(local_dir)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(local_dir)
except Exception:
    print("Model not found in your local machine. Dowloading it from huggingface")
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=local_dir)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, cache_dir=local_dir)

input_text = """Some members of the jury were in tears as the verdict was read around 9:30 pm Friday. 
They had earlier indicated to the judge in two separate messages that they were deadlocked on the charge of using excessive force on Taylor but chose to continue deliberating. 
The six man, six woman jury deliberated for more than 20 hours over three days. Hankison fired 10 shots into Taylor’s glass door and windows during the raid, but didn’t hit anyone. 
Some shots flew into a next-door neighbor’s adjoining apartment.
A separate jury deadlocked on federal charges against Hankison last year, while in 2022, a jury acquitted Hankison on state charges of wanton endangerment."""

inputs = processor(text=input_text, return_tensors="pt")

with torch.no_grad():
    speech = model.generate(**inputs)

sample_rate = 22058
sd.play(speech.numpy(), samplerate = sample_rate)
sd.wait()

print("completed")