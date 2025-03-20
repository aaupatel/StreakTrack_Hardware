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
            attendance_marked[recognized_student_id] = datetime.now().isoformat()  # Convert to string
            attendance_data = {
                "deviceId": load_config().get("deviceId"),
                "studentId": recognized_student_id,
                "timestamp": utils.format_timestamp(datetime.fromisoformat(attendance_marked[recognized_student_id])) #convert back to datetime for timestamp
            }
            if website_websocket:
                await website_websocket.send(json.dumps({"type": "attendance", "attendanceData": attendance_data}))
            else:
                print("Error: WebSocket connection not established. Attendance data not sent.")

            utils.save_json("attendance.json", attendance_marked)
            print(f"Attendance marked for student ID: {recognized_student_id}")
            utils.lcd_display(f"{get_student_name(recognized_student_id)}\nAttendance marked")
            await asyncio.sleep(2)
            utils.lcd_welcome()
            utils.blink_led(GREEN_LED_PIN, 1)
        else:
            print(f"Attendance already marked for student ID: {recognized_student_id}")
            utils.lcd_display(f"{get_student_name(recognized_student_id)}\nAlready marked")
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
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    try:
        db_conn = sqlite3.connect('student_faces.db')
        cursor = db_conn.cursor()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return

    while website_websocket:
        frame = picam2.capture_array()
        if frame is None:
            print("Error: Could not capture frame.")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Add this line
        face_locations = face_recognition.face_locations(frame_rgb, model="hog") #use frame_rgb
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations) #use frame_rgb

        for face_encoding, face_location in zip(face_encodings, face_locations):
            try:
                cursor.execute("SELECT student_id, encoding FROM students")
                results = cursor.fetchall()

                best_match = None
                min_distance = 1.0

                for student_id, encoded_data in results:
                    known_encoding = pickle.loads(encoded_data)
                    distance = face_recognition.face_distance([known_encoding], face_encoding)[0]

                    if distance < min_distance and distance < 0.6:
                        min_distance = distance
                        best_match = student_id

                if best_match:
                    print(f"Recognized: {best_match}")
                    utils.lcd_display("Please wait...")
                    await mark_attendance(best_match)
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, best_match, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
                    await asyncio.sleep(1)
                    GPIO.output(GREEN_LED_PIN, GPIO.LOW)
                else:
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                GPIO.output(YELLOW_LED_PIN, GPIO.HIGH)
            except sqlite3.Error as e:
                print(f"Database query error: {e}")
            except Exception as e:
                print(f"recognition error: {e}")

        else:
            GPIO.output(YELLOW_LED_PIN, GPIO.LOW)

        # cv2.imshow("Camera Feed", frame) #keep this to view the camera feed otherwise Comment out.
        # cv2.waitKey(1) #keep this to view the camera feed otherwise Comment out.

        if streaming_active:
            frame_base64 = stream_utils.encode_frame(frame)
            try:
                await website_websocket.send(json.dumps({"type": "live_stream", "frame": frame_base64}))
            except websockets.exceptions.ConnectionClosed:
                print("Error: WebSocket connection closed while streaming.")
                break

        await asyncio.sleep(0.05)
    picam2.stop() # stop the camera.
    # cv2.destroyAllWindows() #keep this to view the camera feed otherwise Comment out.
    db_conn.close()

def get_student_name(student_id):
    for student in students:
        if student["_id"] == student_id:
            return student["name"]
    return "Unknown"

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