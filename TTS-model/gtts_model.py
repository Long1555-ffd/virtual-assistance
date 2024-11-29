from gtts import gTTS
import pygame
import io

def speak_text(text):
    # Generate TTS and save to a memory buffer
    tts = gTTS(text=text, lang='en', slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)  # Reset the file pointer to the beginning
    
    # Initialize pygame mixer
    pygame.init()
    pygame.mixer.init()
    
    # Load the MP3 from the buffer
    pygame.mixer.music.load(mp3_fp, "mp3")
    
    # Play the audio
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait for playback to finish
        pass


text = """Pete Hegseth, Donald Trump’s defense secretary pick, was among several cabinet nominees and appointees of the president-elect’s incoming administration who were targeted with bomb threats and so-called “swatting” on Wednesday, the Guardian has learned.

Elise Stefanik, a Republican congresswoman of New York and Trump’s pick for US ambassador to the United Nations, who has emerged as a hard-right loyalist of Trump in the last few years, was the subject of a bomb threat, her office said.

The home of Howard Lutnick, Trump’s choice for commerce secretary and part of his transition team, was threatened, the Bronx outlet News 12 reported. And Lee Zeldin, the Environmental Protection Agency pick, saw his Long Island home threatened, News 12 in Long Island also reported.

Zeldin later posted on X saying: “A pipe bomb threat targeting me and my family at our home today was sent in with a pro-Palestinian themed message.” He said they were not at home and were trying to find out more."""
speak_text(text)
