import asyncio
import aiohttp
import face_recognition
import numpy as np
import pickle
import sqlite3
import cv2
from picamera2 import Picamera2
import time

async def download_image(session, url):
    async with session.get(url) as response:
        return await response.read()

async def encode_and_store_students(students):
    """Encodes student images and stores encodings in the database."""
    db_conn = sqlite3.connect('student_faces.db')
    db_conn.execute("CREATE TABLE IF NOT EXISTS students (student_id TEXT PRIMARY KEY, encoding BLOB)")
    await process_students(students, db_conn)
    db_conn.close()

async def encode_and_store(student, db_conn):
    async with aiohttp.ClientSession() as session:
        encodings = []
        for image_url in student['images']:
            try:
                image_data = await download_image(session, image_url)
                image_array = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is not None:
                    face_enc = face_recognition.face_encodings(image)
                    if face_enc:
                        encodings.append(face_enc[0])
            except Exception as e:
                print(f"Error processing {image_url}: {e}")

        if encodings:
            mean_encoding = np.mean(encodings, axis=0)
            # Use INSERT OR REPLACE instead of INSERT
            db_conn.execute("INSERT OR REPLACE INTO students (student_id, encoding) VALUES (?, ?)",
                            (student['_id'], pickle.dumps(mean_encoding)))
            db_conn.commit()
        else:
            print(f"No valid faces for {student['name']}")
async def process_students(students, db_conn):
    tasks = [encode_and_store(student, db_conn) for student in students]
    await asyncio.gather(*tasks)

def setup_camera(width=800, height=600, framerate=15):
    """Sets up the camera."""
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (width, height)})
    picam2.configure(config)
    picam2.start()
    return picam2

def recognize_face(frame):
    """Recognizes a face in the frame and measures detection and recognition time."""
    start_time = time.time()  # Start timing

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    normalized_image = cv2.equalizeHist(gray_image)

    detection_start = time.time()
    face_locations = face_recognition.face_locations(normalized_image, 1, model="hog")
    detection_end = time.time()
    detection_time = detection_end - detection_start

    face_locations = [(top * 2, right * 2, bottom * 2, left * 2) for (top, right, bottom, left) in face_locations]

    if face_locations:
        recognition_start = time.time()
        face_encoding = face_recognition.face_encodings(frame_rgb, face_locations, 1, model="large")
        recognition_end = time.time()
        recognition_time = recognition_end - recognition_start

        if face_encoding:
            face_encoding = face_encoding[0]
            db_conn = sqlite3.connect('student_faces.db')
            cursor = db_conn.cursor()
            cursor.execute("SELECT student_id, encoding FROM students")
            results = cursor.fetchall()
            db_conn.close()

            best_match = None
            min_distance = 0.6  # Adjust this threshold

            for student_id, encoded_data in results:
                known_encoding = pickle.loads(encoded_data)
                distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                if distance < min_distance:
                    min_distance = distance
                    best_match = student_id
            if best_match:
                # print(f"Detection Time: {detection_time:.4f} seconds")
                print(f"Recognition Time: {recognition_time:.4f} seconds")
                return best_match
    # else:
    #     # print(f"Detection Time: {detection_time:.4f} seconds")
    #     # print("No face detected.")

    # return None

def get_student_name(student_id, students):
    """Gets the student name from the student ID."""
    for student in students:
        if student["_id"] == student_id:
            return student["name"]
    return "Unknown"
