import asyncio
import websockets
import json
import cv2
import RPi.GPIO as GPIO
from datetime import datetime
# import stream_utils
from picamera2 import Picamera2, Preview
import numpy as np
import utils
import sqlite3
import face_recognition
import pickle
import os
import face_utils  # Import the face utility module
import stream_utils  # Import the streaming utility module

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

    uri = f"{website_url}/api/ws?deviceId={dev_id}&organizationId={org_id}&isHardware=true"

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
    """Fetches student data from the server, encodes images, and saves it with error handling."""
    global students, website_websocket
    try:
        if website_websocket:
            response = await website_websocket.recv()
            data = json.loads(response)
            if "students" in data:
                students = data["students"]
                await face_utils.encode_and_store_students(students) #Encode images and store.
                # utils.save_json("student_data.json", students)
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
        if recognized_student_id not in attendance_marked:
            timestamp = datetime.now()
            attendance_marked[recognized_student_id] = datetime.now().isoformat()  # Convert to string
            student_data = {
                "name": face_utils.get_student_name(recognized_student_id, students),
                "deviceId": load_config().get("deviceId"),
                "studentId": recognized_student_id,
                "timestamp": utils.format_timestamp(datetime.fromisoformat(attendance_marked[recognized_student_id]))  # convert back to datetime for timestamp
            }
            if website_websocket:
                await website_websocket.send(json.dumps({"type": "attendance", "student": student_data}))
            else:
                print("Error: WebSocket connection not established. Attendance data not sent.")

            utils.save_json("attendance.json", attendance_marked)
            print(f"Attendance marked for student ID: {recognized_student_id}")
            # Corrected line: Use face_utils.get_student_name
            utils.lcd_display(f"{face_utils.get_student_name(recognized_student_id, students)}\nAttendance marked")
            await asyncio.sleep(2)
            utils.lcd_welcome()
            utils.blink_led(GREEN_LED_PIN, 1)
        else:
            print(f"Attendance already marked for student ID: {recognized_student_id}")
            # Corrected line: Use face_utils.get_student_name
            utils.lcd_display(f"{face_utils.get_student_name(recognized_student_id, students)}\nAlready marked")
            await asyncio.sleep(2)
            utils.lcd_welcome()
            utils.blink_led(YELLOW_LED_PIN, 1)
    except Exception as e:
        print(f"Error marking attendance: {e}")
        utils.lcd_display("Error")
        await asyncio.sleep(2)
        utils.lcd_welcome()

async def camera_loop():
    """Main camera loop for face detection, recognition, and streaming."""
    picam2 = face_utils.setup_camera(width=640, height=480, framerate=15) #Setup camera.
    try:
        while website_websocket:
            frame = picam2.capture_array()
            recognized_student_id = face_utils.recognize_face(frame)
            
            if recognized_student_id:
                await mark_attendance(recognized_student_id)
            if streaming_active:
                frame_base64 = stream_utils.encode_frame(frame)
                try:
                    # print("sending frame") #added log
                    await website_websocket.send(json.dumps({"type": "live_stream", "frame": frame_base64}))
                    # print("frame sent") #added log
                except websockets.exceptions.ConnectionClosed:
                    print("Error: WebSocket connection closed while streaming.")
                    break
                except Exception as e:
                    print(f"error sending frame: {e}")

                # Display the frame locally using OpenCV (optional, remove if not needed)
                # cv2.imshow("Camera Feed", frame)
                # cv2.waitKey(1)

            await asyncio.sleep(0.03)
        
        # cv2.destroyAllWindows()
        picam2.stop()
    except Exception as e:
        print(f"Camera Loop error: {e}")

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
            await camera_loop()
        GPIO.output(RED_LED_PIN, GPIO.LOW)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    asyncio.run(main())