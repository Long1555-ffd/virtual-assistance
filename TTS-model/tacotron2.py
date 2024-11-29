# import torchaudio
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN
from scipy.io.wavfile import write
import numpy as np
# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

text = """Doctors are hailing a new way to treat serious asthma and chronic obstructive pulmonary disease attacks that marks the first breakthrough for 50 years and could be a “gamechanger” for patients.
A trial found offering patients an injection was more effective than the current care of steroid tablets, and cuts the need for further treatment by 30%.
The results, published in the Lancet Respiratory Medicine journal, could be transformative for millions of people with asthma and COPD around the world."""

text2= """Tell me a bit about who you are and how you can assist me to be my virtual assistance?"""
# Running the TTS
mel_output, mel_length, alignment = tacotron2.encode_text(text2)

print("encoded text:", mel_output)
# Running Vocoder (spectrogram-to-waveform)
waveforms = hifi_gan.decode_batch(mel_output)
print("spectrogram:", waveforms)

# Save the waverform
# torchaudio.save('example_TTS.wav',waveforms.squeeze(1), 22050, format="wav")

# convert pytorch tensor to numpy arrays
waveform_numpy = waveforms.squeeze(1).numpy().flatten()
print("squeeze the waveform and convert it to numpy array:", waveform_numpy)
print(f"min and max values: {waveform_numpy.min()} {waveform_numpy.max()}" )
waveform_numpy_int16 = np.int16(np.round(waveform_numpy*32767))
print(f"after normalization: {waveform_numpy_int16.min()}, {waveform_numpy_int16.max()}")
write("example-TTS.wav", 22050, waveform_numpy_int16)