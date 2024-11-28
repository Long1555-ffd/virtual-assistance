from gtts import gTTS
import os

def speak_text(text):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")  # Play audio (ensure mpg321 is installed)

text = """There are still over 50 days left until Donald Trump takes office, but he’s already laid the ground for a trade war that could shake the global economy.
Trump announced on Monday that he will sign an executive order placing a 25% tariff on all imports from Canada and Mexico, along with an additional 10% tariff on imports from China, in purported retaliation for drugs and migrants crossing US borders.
"""
speak_text(text)
