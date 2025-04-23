import asyncio
import websockets
import json
import cv2
import RPi.GPIO as GPIO
from datetime import datetime
from picamera2 import Picamera2, Preview
import numpy as np
import utils
import sqlite3
import face_recognition
import pickle
import os
import face_utils
import stream_utils

# GPIO Setup
GREEN_LED_PIN = 32
YELLOW_LED_PIN = 36
BLUE_LED_PIN = 38
RED_LED_PIN = 40

GPIO.setmode(GPIO.BOARD)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(YELLOW_LED_PIN, GPIO.OUT)
GPIO.setup(BLUE_LED_PIN, GPIO.OUT)
GPIO.setup(RED_LED_PIN, GPIO.OUT)

# Global Variables
website_websocket = None
students = []
attendance_marked = {}
streaming_active = False
face_detected = False
frame_queue = asyncio.Queue(maxsize=10) # Create a queue for frame storage
latest_frame = None

# Config File
CONFIG_FILE = "config.json"

def load_config():
    """Loads configuration from config.json with error handling."""
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        if not config:
            raise ValueError("Config file is empty or invalid.")
        return config
    except FileNotFoundError:
        print(f"Error: Config file '{CONFIG_FILE}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Config file '{CONFIG_FILE}' is not valid JSON.")
        return None
    except ValueError as e:
        print(f"Error loading config: {e}")
        return None

async def connect_websocket():
    """Connects to the WebSocket server with error handling."""
    global website_websocket, students
    config = load_config()
    if not config:
        return

    website_url = config.get("website_url")
    dev_id = config.get("deviceId")
    org_id = config.get("organizationId")
    if not all([website_url, dev_id, org_id]):
        print("Error: Incomplete configuration in config.json.")
        return

    uri = f"{website_url}/api/ws?deviceId={dev_id}&organizationId={org_id}&isHardware=true"

    try:
        website_websocket = await websockets.connect(uri)
        GPIO.output(BLUE_LED_PIN, GPIO.HIGH)
        print("Connected to StreakTrack")
        attendance_marked = {}
        utils.save_json("attendance.json", {})
        return True #connection successful
    except websockets.exceptions.ConnectionRefusedError:
        print("Error: Connection refused by server. Check URL or server status.")
        GPIO.output(RED_LED_PIN, GPIO.HIGH)
        GPIO.output(BLUE_LED_PIN, GPIO.LOW)
        return False
    except websockets.exceptions.InvalidURI:
        print("Error: Invalid WebSocket URI in config.json.")
        GPIO.output(RED_LED_PIN, GPIO.HIGH)
        GPIO.output(BLUE_LED_PIN, GPIO.LOW)
        return False
    except Exception as e:
        print(f"Error connecting to StreakTrack: {e}")
        GPIO.output(RED_LED_PIN, GPIO.HIGH)
        GPIO.output(BLUE_LED_PIN, GPIO.LOW)
        return False

async def fetch_students():
    """Fetches student data from the server, encodes images, and saves it with error handling."""
    global students, website_websocket
    try:
        if website_websocket:
            if os.path.exists('student_faces.db'):
                os.remove('student_faces.db')
                print("Old student_faces.db deleted.")
            response = await website_websocket.recv()
            data = json.loads(response)
            if "students" in data:
                students = data["students"]
                await face_utils.encode_and_store_students(students) #Encode images and store in the database.
                print("Student data fetched, encoded, and stored.")
                utils.lcd_display("Students\nreceived")
                await asyncio.sleep(2)
                utils.lcd_welcome()
            else:
                print("Error: No student data received from server.")
        else:
            print("Error: WebSocket connection not established.")
    except websockets.exceptions.ConnectionClosed:
        print("Error: WebSocket connection closed while fetching students.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON received from server.")
    except Exception as e:
        print(f"Error fetching student data: {e}")

async def mark_attendance(recognized_student_id):
    """Marks attendance and sends data to the server with error handling."""
    global attendance_marked, website_websocket
    try:
        student_name = face_utils.get_student_name(recognized_student_id, students)
        enrollment_no = face_utils.get_enrollment_no(recognized_student_id, students)
        device_id = load_config().get("deviceId")

        timestamp = datetime.now()

        if recognized_student_id not in attendance_marked:
            attendance_marked[recognized_student_id] = timestamp.isoformat()  # Save ISO format

            student_data = {
                "name": student_name,
                "enrollmentNo": enrollment_no,
                "deviceId": device_id,
                "studentId": recognized_student_id,
                "timestamp": utils.format_timestamp(timestamp)
            }

            if website_websocket:
                await website_websocket.send(json.dumps({"type": "attendance", "student": student_data}))
            else:
                print("Error: WebSocket connection not established. Attendance data not sent.")

            utils.save_json("attendance.json", attendance_marked)
            print(f"Attendance marked for student ID: {recognized_student_id}")
            utils.lcd_display(f"{student_name}\nAttendance marked")
            await asyncio.sleep(2)
            utils.lcd_welcome()
            utils.blink_led(GREEN_LED_PIN, 1)
        else:
            print(f"Attendance already marked for student ID: {recognized_student_id}")
            utils.lcd_display(f"{student_name}\nAlready marked")
            await asyncio.sleep(2)
            utils.lcd_welcome()
            utils.blink_led(GREEN_LED_PIN, 1)
            utils.blink_led(YELLOW_LED_PIN, 1)

            # Send detected event even if attendance was already marked
            student_data = {
                "name": student_name,
                "enrollmentNo": enrollment_no,
                "deviceId": device_id,
                "studentId": recognized_student_id,
                "timestamp": utils.format_timestamp(timestamp)
            }
            if website_websocket:
                await website_websocket.send(json.dumps({"type": "detected", "student": student_data}))
            else:
                print("Error: WebSocket connection not established. Detected data not sent.")

    except Exception as e:
        print(f"Error marking attendance: {e}")
        utils.lcd_display("Error")
        await asyncio.sleep(2)
        utils.lcd_welcome()

async def capture_frame(picam2):
    """Capture frames continuously and push them into the queue."""
    global latest_frame
    while True:
        frame = picam2.capture_array()
        latest_frame = frame.copy()  # Clone to prevent conflict
        if frame_queue.full():
            await frame_queue.get()
        await frame_queue.put(frame)  # Still allow queue for streaming
        await asyncio.sleep(0)

async def streaming_loop():
    """Continuously stream frames."""
    while True:
        if streaming_active and not frame_queue.empty():
            frame = await frame_queue.get()
            frame_base64 = stream_utils.encode_frame(frame)
            try:
                await website_websocket.send(json.dumps({
                    "type": "live_stream",
                    "frame": frame_base64
                }))
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed while streaming.")
                break
            except Exception as e:
                print(f"Error sending stream frame: {e}")
        await asyncio.sleep(0)  # Run as fast as possible

async def face_recognition_loop():
    """Run face recognition continuously with a delay of 0.5 seconds."""
    global face_detected
    global latest_frame
    processing_face = False
    while True:
        if latest_frame is not None and not processing_face:
            frame = latest_frame.copy()
            recognized_student_id = face_utils.recognize_face(frame)
            if recognized_student_id is not None:
                processing_face = True
                GPIO.output(YELLOW_LED_PIN, GPIO.HIGH)
                GPIO.output(GREEN_LED_PIN, GPIO.LOW)
                utils.lcd_display("Please Wait...")
                if recognized_student_id == "Unknown":
                    utils.lcd_display("Unknown Person")
                    GPIO.output(RED_LED_PIN, GPIO.HIGH)
                    await asyncio.sleep(1)
                    utils.lcd_welcome()
                else:
                    await mark_attendance(recognized_student_id)
                processing_face = False
            else:
                GPIO.output(YELLOW_LED_PIN, GPIO.LOW)
                GPIO.output(GREEN_LED_PIN, GPIO.LOW)
                GPIO.output(RED_LED_PIN, GPIO.LOW)
        await asyncio.sleep(0.5)

async def websocket_message_handler():
    """Handles incoming WebSocket messages with error handling."""
    global streaming_active
    while website_websocket:
        try:
            message = await website_websocket.recv()
            data = json.loads(message)
            if data.get("type") == "start_stream":
                streaming_active = True
                print("Streaming started.")
            elif data.get("type") == "stop_stream":
                streaming_active = False
                print("Streaming stopped.")
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed.")
            break
        except json.JSONDecodeError:
            print("Received invalid JSON message.")
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")

async def main():
    """Main function to start the application."""
    try:
        GPIO.output(RED_LED_PIN, GPIO.HIGH)
        if await connect_websocket():
            utils.lcd_display("Connected")
            await asyncio.sleep(2)
            utils.lcd_welcome()
            await fetch_students()
            asyncio.create_task(websocket_message_handler())

            picam2 = face_utils.setup_camera(width=640, height=480)
            asyncio.create_task(capture_frame(picam2))
            asyncio.create_task(streaming_loop())
            asyncio.create_task(face_recognition_loop())

            while True:
                await asyncio.sleep(1)
        GPIO.output(RED_LED_PIN, GPIO.LOW)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    asyncio.run(main())