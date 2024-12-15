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