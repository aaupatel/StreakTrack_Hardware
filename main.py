import asyncio
import websockets
import json
import cv2
import base64
import RPi.GPIO as GPIO
from datetime import datetime
import face_utils
import stream_utils
import utils
import os

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

# Config File
CONFIG_FILE = "config.json"

def load_config():
    """Loads configuration from config.json with error handling."""
    try:
        config = utils.load_json(CONFIG_FILE)
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

    uri = f"{website_url}/api/ws?deviceId={dev_id}&organizationId={org_id}"

    try:
        website_websocket = await websockets.connect(uri)
        GPIO.output(BLUE_LED_PIN, GPIO.HIGH)
        print("Connected to StreakTrack")
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
    """Fetches student data from the server and saves it with error handling."""
    global students, website_websocket
    try:
        if website_websocket:
            response = await website_websocket.recv()
            data = json.loads(response)
            if "students" in data:
                students = data["students"]
                utils.save_json("student_data.json", students)
                print("Student data fetched and saved.")
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

async def mark_attendance(recognized_student):
    """Marks attendance and sends data to the server with error handling."""
    global attendance_marked, website_websocket
    try:
        student_id = recognized_student["_id"]
        if student_id not in attendance_marked:
            attendance_marked[student_id] = datetime.now()
            attendance_data = {
                "deviceId": load_config().get("deviceId"),
                "studentId": student_id,
                "timestamp": utils.format_timestamp(attendance_marked[student_id])
            }
            if website_websocket:
                await website_websocket.send(json.dumps({"type": "attendance", "attendanceData": attendance_data}))
            else:
                print("Error: WebSocket connection not established. Attendance data not sent.")

            utils.save_json("attendance.json", attendance_marked)
            print(f"Attendance marked for {recognized_student['name']}")
            utils.blink_led(GREEN_LED_PIN, 1)
        else:
            print(f"Attendance already marked for {recognized_student['name']}")
            utils.blink_led(YELLOW_LED_PIN, 1)
    except Exception as e:
        print(f"Error marking attendance: {e}")

async def camera_loop():
    """Main camera loop for face detection, recognition, and streaming."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while website_websocket:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            # break

        faces = face_utils.detect_faces(frame)
        for (x, y, w, h) in faces:
            GPIO.output(YELLOW_LED_PIN, GPIO.HIGH)
            # recognized_student = face_utils.recognize_faces(face_image, students)
            recognized_student = face_utils.recognize_faces(frame[y:y+h, x:x+w], students)
            if recognized_student:
                await mark_attendance(recognized_student)

            GPIO.output(YELLOW_LED_PIN, GPIO.LOW)

        if streaming_active:
            frame_base64 = stream_utils.encode_frame(frame)
            try:
                await website_websocket.send(json.dumps({"type": "live_stream", "frame": frame_base64}))
            except websockets.exceptions.ConnectionClosed:
                print("Error: WebSocket connection closed while streaming.")
                break

        await asyncio.sleep(0.1)
    cap.release()

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
          await fetch_students()
          asyncio.create_task(websocket_message_handler())
          await camera_loop()
        GPIO.output(RED_LED_PIN, GPIO.LOW)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    asyncio.run(main())