import os
import sys
import subprocess
import speech_recognition as sr
import pyttsx3
import pywhatkit
import wikipedia
import time
from datetime import datetime, timedelta
import openai

# Cấu hình OpenAI API
# openai.api_key = 'your-openai-api-key'

# Chuyển mã hóa console sang UTF-8
os.system('chcp 65001')
sys.stdout.reconfigure(encoding='utf-8')

# Khởi tạo recognizer và engine text-to-speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()

def talk(text):
    engine.say(text)
    engine.runAndWait()

def take_command():
    try:
        with sr.Microphone() as source:
            print("Đang nghe...")
            voice = recognizer.listen(source)
            command = recognizer.recognize_google(voice, language='en-US')
            command = command.lower()
            if 'sarah' in command:
                command = command.replace('hey sarah', '')
                command = command.replace('ok sarah', '')
                print(command)
                return command
    except Exception as e:
        print(e)
        return "Please repeat the command"
    return ""

def turn_on_light():
    # Mô phỏng bật đèn
    print("Turn on the light")
    talk("Turn on the light")

def turn_off_light():
    # Mô phỏng tắt đèn
    print("Turn off the light")
    talk("Turn off the loght")

def open_application(app_name):
    # Hàm để mở phần mềm
    if app_name == 'notepad':
        subprocess.run(['notepad.exe'])
        talk("Opening Notepad")
    elif app_name == 'calculator':
        subprocess.run(['calc.exe'])
        talk("Opening the calculator")
    else:
        talk("There is an error in opening" + app_name)

def tell_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    talk(f"The current time is {current_time}")
    print(f"The current time is {current_time}")

def tell_date():
    now = datetime.now()
    current_date = now.strftime("%d/%m/%Y")
    talk(f"The data today is {current_date}")
    print(f"The data today is {current_date}")

def tell_day():
    now = datetime.now()
    day_of_week = now.strftime("%A")
    talk(f"Date of the week {day_of_week}")
    print(f"Date of the week {day_of_week}")

def get_specific_date_info(date_str):
    try:
        if 'yesterday' in date_str:
            date_obj = datetime.now() - timedelta(days=1)
        elif 'tomorrow' in date_str:
            date_obj = datetime.now() + timedelta(days=1)
        else:
            date_obj = datetime.strptime(date_str, "%d/%m/%Y")
        
        day_of_week = date_obj.strftime("%A")
        talk(f" {date_obj.strftime('%d/%m/%Y')} {day_of_week}")
        print(f" {date_obj.strftime('%d/%m/%Y')} {day_of_week}")
    except ValueError:
        talk("Invalid datetime, please try again")
        print("Invalid datetime, please try again")

def get_openai_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print("Lỗi khi gọi OpenAI API: " + str(e))
        return "Có lỗi xảy ra khi truy vấn OpenAI API."

def run_sarah():
    command = take_command()
    if command:
        print(command)
        if 'play music' in command:
            song = command.replace('play music', '')
            talk('Playing the song' + song)
            pywhatkit.playonyt(song)
        elif 'Play a random song' in command:
            talk('Playing a random song on youtube')
            pywhatkit.playonyt('lofi chill')
        elif 'tell me about' in command:
            person = command.replace('tell me about', '')
            info = wikipedia.summary(person, sentences=1)
            talk(info)
        elif 'turn on the light' in command:
            turn_on_light()
        elif 'turn off the light' in command:
            talk("Yes, turning the light off")
            time.sleep(5)
            turn_off_light()
        elif 'open' in command:
            app_name = command.replace('open', '').strip()
            open_application(app_name)
        elif 'The current time' in command:
            tell_time()
        elif 'What is the date today' in command:
            tell_date()
        elif 'date of the week' in command:
            tell_day()
        elif 'ngày' in command and 'là thứ mấy' in command:
            date_str = command.replace('ngày', '').replace('là thứ mấy', '').strip()
            get_specific_date_info(date_str)
        elif 'ngày' in command and 'là ngày mấy' in command:
            date_str = command.replace('ngày', '').replace('là ngày mấy', '').strip()
            get_specific_date_info(date_str)
        elif 'what is the date yesterday' in command:
            get_specific_date_info('yesterday')
        elif 'what is the date tomorrow' in command:
            get_specific_date_info('tomorrow')
        elif 'help me with' in command:
            prompt = command.replace('help me with', '').strip()
            response = get_openai_response(prompt)
            talk(response)
        else:
            talk('try again with your command')

# Lắng nghe liên tục
while True:
    run_sarah()
