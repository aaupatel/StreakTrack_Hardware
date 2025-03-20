import json
import RPi.GPIO as GPIO
from datetime import datetime
import time
import cv2
import base64
from RPLCD.i2c import CharLCD

# LCD setup (initialize globally)
try:
    lcd = CharLCD('PCF8574', address=0x27, cols=20, rows=4) #20 columns and 4 rows.
    lcd.clear()
    lcd.write_string("Welcome to StreakTrack")
except Exception as e:
    print(f"LCD initialization failed: {e}")
    lcd = None

def load_json(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f)

def blink_led(pin, duration):
    GPIO.output(pin, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(pin, GPIO.LOW)

def format_timestamp(timestamp):
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def encode_frame(frame):
    _, frame_encoded = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(frame_encoded).decode('utf-8')
    return frame_base64

def lcd_display(message):
    if lcd:
        lcd.clear()
        lines = wrap_text(message, 20) #20 characters per line
        for i, line in enumerate(lines):
            if i < 4:  # Display up to 4 lines
                lcd.cursor_pos = (i, 0)
                lcd.write_string(line)

def lcd_welcome():
    if lcd:
        lcd.clear()
        lcd.write_string("Welcome to StreakTrack")

def wrap_text(text, line_length):
    """Wraps text to fit within a given line length."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= line_length:
            if current_line:
                current_line += " "
            current_line += word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines