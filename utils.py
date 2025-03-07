import json
import RPi.GPIO as GPIO
from datetime import datetime
import time

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