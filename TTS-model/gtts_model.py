from gtts import gTTS
import os

def speak_text(text):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")  # Play audio (ensure mpg321 is installed)

text = "Hello, how are you today?"
speak_text(text)
