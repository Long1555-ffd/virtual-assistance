import edge_tts
import os
import numpy as np
import sounddevice as sd
from io import BytesIO
import wave
async def text_to_speech_stream(text,voice="en-US-AriaNeural", rate="+0%"):
    # this is for edge-tts initialization
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    
    # generate the TTS output as a stream
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]

def play_audio_stream(audio_stream):
    for audio_chunk in audio_stream:
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        sd.play(audio_data, samplerate=24000)
        sd.wait() # wait for each chunk to finish the playing

async def main(text):
    audio_stream = []
    async for audio_chunk in text_to_speech_stream(text):
        audio_stream.append(audio_chunk)

    play_audio_stream(audio_stream)
    
import asyncio

text= """There are still over 50 days left until Donald Trump takes office, but he’s already laid the ground for a trade war that could shake the global economy.
Trump announced on Monday that he will sign an executive order placing a 25% tariff on all imports from Canada and Mexico, along with an additional 10% tariff on imports from China, in purported retaliation for drugs and migrants crossing US borders.
"""

asyncio.run(main(text))