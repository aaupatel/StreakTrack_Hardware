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
    """Downloads an image from a given URL asynchronously."""
    async with session.get(url) as response:
        return await response.read()

async def encode_and_store_students(students):
    """Encodes face embeddings for all students and stores them in a database."""
    db_conn = sqlite3.connect('student_faces.db')
    db_conn.execute("DROP TABLE IF EXISTS students") # Drop the table if it exists, then create it to ensure a fresh start
    db_conn.execute("CREATE TABLE IF NOT EXISTS students (student_id TEXT PRIMARY KEY, encoding BLOB)") # Create a table to store student IDs and their face encodings (as BLOB data)
    await process_students(students, db_conn) # Process each student to download images, encode faces, and store in the database
    db_conn.close()

async def encode_and_store(student, db_conn):
    """Downloads images for a single student, encodes detected faces, and stores the encodings in the database."""
    async with aiohttp.ClientSession() as session:
        all_face_encodings = []
        if student['images']:
            for image_url in student['images']:
                try:
                    image_data = await download_image(session, image_url) # Download the image data from the URL
                    image_array = np.frombuffer(image_data, np.uint8) # Convert the downloaded image data to a NumPy array
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR) # Decode the NumPy array into an OpenCV image object
                    if image is not None:
                        face_encs = face_recognition.face_encodings(image) # Detect face locations in the image using the HOG model (fast but less accurate than CNN)
                        if face_encs:
                            all_face_encodings.extend(face_encs) # Extend the list with all face encodings found in the current image
                except Exception as e:
                    print(f"Error processing {image_url}: {e}")
        else:
            print(f"No images found for {student['name']}")

        if all_face_encodings:
            db_conn.execute("INSERT OR REPLACE INTO students (student_id, encoding) VALUES (?, ?)",  # Store all encoded faces associated with the student's ID in the database
                            (student['_id'], pickle.dumps(all_face_encodings)))
            db_conn.commit()
        else:
            print(f"No valid faces found for {student['name']}")

async def process_students(students, db_conn):
    """Asynchronously processes a list of students to encode and store their face embeddings."""
    tasks = [encode_and_store(student, db_conn) for student in students] # Create a list of tasks, where each task is to encode and store a single student's faces
    await asyncio.gather(*tasks) # Run all the encoding and storing tasks concurrently

def setup_camera(width=640, height=480, framerate=15):
    """Sets up and initializes the Raspberry Pi camera."""
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (width, height)})
    picam2.configure(config)
    picam2.start()
    return picam2

def recognize_face(frame):
    """Recognizes a face in the input frame by comparing it to the stored face embeddings."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the BGR frame (OpenCV's default) to RGB (face_recognition's requirement)
    gray_image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)  # Convert the RGB frame to grayscale for histogram equalization
    normalized_image = cv2.equalizeHist(gray_image)  # Apply histogram equalization to improve contrast in the grayscale image, aiding face detection
    # Detect the locations of faces in the normalized grayscale image using the HOG model
    face_locations = face_recognition.face_locations(normalized_image, model="hog") # Using hog for speed

    if face_locations:
        # Encode the detected faces in the RGB frame to get their feature embeddings
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        if face_encodings:
            # Connect to the SQLite database containing the stored student face embeddings
            db_conn = sqlite3.connect('student_faces.db')
            cursor = db_conn.cursor()
            # Retrieve all student IDs and their corresponding face embeddings from the database
            cursor.execute("SELECT student_id, encoding FROM students")
            results = cursor.fetchall()
            # Close the database connection
            db_conn.close()

            for face_encoding in face_encodings:
                best_match = None
                min_distance = 0.5  # Adjust threshold as needed (very similar enough < 0.5 < not similar enough)

                for student_id, encoded_data in results:
                    # Load the stored face embeddings for the current student from the BLOB data
                    known_encodings = pickle.loads(encoded_data)
                    # Iterate through all the stored encodings for the current student (a student have multiple images)
                    for known_encoding in known_encodings:
                        # Calculate the Euclidean distance between the detected face's encoding and the known encoding
                        distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                        # If the calculated distance is less than the current minimum distance, update the best match
                        if distance < min_distance:
                            min_distance = distance
                            best_match = student_id

                # If a best match is found (distance is below the threshold), return the student ID
                if best_match:
                    return best_match
                else:
                    # If no match is found below the threshold, return "Unknown"
                    return "Unknown"

    # If no faces are detected in the frame, return None
    return None

def get_student_name(student_id, students):
    """Retrieves the name of a student given their ID from the list of student data."""
    for student in students:
        if student["_id"] == student_id:
            return student["name"]
    return "Unknown"