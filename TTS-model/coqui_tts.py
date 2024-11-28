from TTS.api import TTS

model_name ="tts_model/en/ljspeech/tacotron2-DDC"

tts = TTS(model_name=model_name)

text = "Hello, this is a test of the Coqui TTS system"

tts.tts_to_file(text=text, file_name="output.wav")

print("Audio saved")