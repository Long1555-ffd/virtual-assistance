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
            command = recognizer.recognize_google(voice, language='vi-VN')
            command = command.lower()
            if 'sarah' in command:
                command = command.replace('hey sarah', '')
                command = command.replace('ok sarah', '')
                print(command)
                return command
    except Exception as e:
        print("Lỗi: " + str(e))
        return "Không nghe rõ"
    return ""

def turn_on_light():
    # Mô phỏng bật đèn
    print("Đèn đã bật")
    talk("Vâng thưa ông, tôi đang bật đèn")

def turn_off_light():
    # Mô phỏng tắt đèn
    print("Đèn đã tắt")
    talk("Tôi đang tắt đèn")

def open_application(app_name):
    # Hàm để mở phần mềm
    if app_name == 'notepad':
        subprocess.run(['notepad.exe'])
        talk("Đang mở Notepad")
    elif app_name == 'calculator':
        subprocess.run(['calc.exe'])
        talk("Đang mở Máy tính")
    else:
        talk("Xin lỗi, tôi không biết cách mở " + app_name)

def tell_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    talk(f"Bây giờ là {current_time}")
    print(f"Bây giờ là {current_time}")

def tell_date():
    now = datetime.now()
    current_date = now.strftime("%d/%m/%Y")
    talk(f"Hôm nay là ngày {current_date}")
    print(f"Hôm nay là ngày {current_date}")

def tell_day():
    now = datetime.now()
    day_of_week = now.strftime("%A")
    talk(f"Hôm nay là {day_of_week}")
    print(f"Hôm nay là {day_of_week}")

def get_specific_date_info(date_str):
    try:
        if 'hôm qua' in date_str:
            date_obj = datetime.now() - timedelta(days=1)
        elif 'ngày mai' in date_str:
            date_obj = datetime.now() + timedelta(days=1)
        else:
            date_obj = datetime.strptime(date_str, "%d/%m/%Y")
        
        day_of_week = date_obj.strftime("%A")
        talk(f"Ngày {date_obj.strftime('%d/%m/%Y')} là {day_of_week}")
        print(f"Ngày {date_obj.strftime('%d/%m/%Y')} là {day_of_week}")
    except ValueError:
        talk("Ngày không hợp lệ, vui lòng thử lại.")
        print("Ngày không hợp lệ, vui lòng thử lại.")

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
        if 'phát nhạc' in command:
            song = command.replace('phát nhạc', '')
            talk('Đang phát ' + song)
            pywhatkit.playonyt(song)
        elif 'phát một bài nhạc ngẫu nhiên' in command:
            talk('Đang phát nhạc ngẫu nhiên trên YouTube')
            pywhatkit.playonyt('nhạc ngẫu nhiên')
        elif 'nói cho tôi biết về' in command:
            person = command.replace('nói cho tôi biết về', '')
            info = wikipedia.summary(person, sentences=1)
            talk(info)
        elif 'bật đèn' in command:
            turn_on_light()
        elif 'tắt đèn' in command:
            talk("Vâng thưa ông, tôi sẽ tắt đèn trong 5 giây nữa")
            time.sleep(5)
            turn_off_light()
        elif 'hãy mở phần mềm' in command:
            app_name = command.replace('hãy mở phần mềm', '').strip()
            open_application(app_name)
        elif 'bây giờ là mấy giờ' in command:
            tell_time()
        elif 'hôm nay là ngày mấy' in command:
            tell_date()
        elif 'hôm nay là thứ mấy' in command:
            tell_day()
        elif 'ngày' in command and 'là thứ mấy' in command:
            date_str = command.replace('ngày', '').replace('là thứ mấy', '').strip()
            get_specific_date_info(date_str)
        elif 'ngày' in command and 'là ngày mấy' in command:
            date_str = command.replace('ngày', '').replace('là ngày mấy', '').strip()
            get_specific_date_info(date_str)
        elif 'hôm qua là ngày mấy' in command:
            get_specific_date_info('hôm qua')
        elif 'ngày mai là ngày mấy' in command:
            get_specific_date_info('ngày mai')
        elif 'giúp tôi với' in command:
            prompt = command.replace('giúp tôi với', '').strip()
            response = get_openai_response(prompt)
            talk(response)
        else:
            talk('Xin vui lòng nói lại lệnh.')

# Lắng nghe liên tục
while True:
    run_sarah()
