import asyncio
import aiohttp
import face_recognition
import numpy as np
import pickle
import sqlite3
import json
import cv2

async def download_image(session, url):
    async with session.get(url) as response:
        return await response.read()

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
            db_conn.execute("INSERT INTO students (student_id, encoding) VALUES (?, ?)",
                            (student['_id'], pickle.dumps(mean_encoding)))
            db_conn.commit()
        else:
            print(f"No valid faces for {student['name']}")

async def process_students(students, db_conn):
    tasks = [encode_and_store(student, db_conn) for student in students]
    await asyncio.gather(*tasks)

try:
    with open("student_data.json", "r") as f:
        students = json.load(f)
except FileNotFoundError:
    print("Error: student_data.json not found.")
    students = []

import sqlite3

db_conn = sqlite3.connect('student_faces.db')
db_conn.execute("CREATE TABLE IF NOT EXISTS students (student_id TEXT PRIMARY KEY, encoding BLOB)")

asyncio.run(process_students(students, db_conn))

db_conn.close()