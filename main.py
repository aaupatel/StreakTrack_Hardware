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

async def connect_websocket():
    global website_websocket, students
    config = utils.load_json(CONFIG_FILE)
    if not config:
        return

    website_url = config.get("website_url")
    dev_id = config.get("deviceId")
    org_id = config.get("organizationId")
    uri = f"{website_url}/api/ws?deviceId={dev_id}&organizationId={org_id}"

    try:
        website_websocket = await websockets.connect(uri)
        GPIO.output(BLUE_LED_PIN, GPIO.HIGH)
        print("Connected to StreakTrack")

        response = await website_websocket.recv()
        data = json.loads(response)
        if "students" in data:
            students = data["students"]
            utils.save_json("student_data.json", students)
            print("Student data fetched and saved.")
        else:
            print("Error fetching student data.")

        asyncio.create_task(websocket_message_handler())
        await camera_loop()
    except Exception as e:
        print(f"Error connecting to StreakTrack: {e}")
        GPIO.output(RED_LED_PIN, GPIO.HIGH)
        GPIO.output(BLUE_LED_PIN, GPIO.LOW)
    finally:
        if website_websocket:
            await website_websocket.close()
            website_websocket = None
        GPIO.output(BLUE_LED_PIN, GPIO.LOW)
        GPIO.output(RED_LED_PIN, GPIO.HIGH)

async def camera_loop():
    cap = cv2.VideoCapture(0)
    while website_websocket:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_utils.detect_faces(frame)
        for (x, y, w, h) in faces:
            GPIO.output(YELLOW_LED_PIN, GPIO.HIGH)
            recognized_student = face_utils.recognize_faces(frame[y:y+h, x:x+w], students)
            if recognized_student:
                student_id = recognized_student["_id"]
                if student_id not in attendance_marked:
                    attendance_marked[student_id] = datetime.now()
                    attendance_data = {
                        "deviceId": utils.load_json(CONFIG_FILE).get("deviceId"),
                        "studentId": student_id,
                        "timestamp": utils.format_timestamp(attendance_marked[student_id])
                    }
                    await website_websocket.send(json.dumps({"type": "attendance", "attendanceData": attendance_data}))
                    utils.save_json("attendance.json", attendance_marked)
                    print(f"Attendance marked for {recognized_student['name']}")
                    utils.blink_led(GREEN_LED_PIN, 1)
                else:
                    print(f"Attendance already marked for {recognized_student['name']}")
                    utils.blink_led(YELLOW_LED_PIN, 1)

            GPIO.output(YELLOW_LED_PIN, GPIO.LOW)

        if streaming_active:
            frame_base64 = stream_utils.encode_frame(frame)
            await website_websocket.send(json.dumps({"type": "live_stream", "frame": frame_base64}))

        await asyncio.sleep(0.1)
    cap.release()

async def websocket_message_handler():
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
            break
        except json.JSONDecodeError:
            print("Received invalid JSON message.")

async def main():
    try:
        GPIO.output(RED_LED_PIN, GPIO.HIGH)
        await connect_websocket()
        GPIO.output(RED_LED_PIN, GPIO.LOW)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    asyncio.run(main())