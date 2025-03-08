import cv2
import face_recognition
import numpy as np
import base64
import requests

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def download_and_encode_image(image_url):
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        _, image_encoded = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(image_encoded).decode('utf-8')
        return image_base64
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {image_url}: {e}")
        return None
    except cv2.error as e:
        print(f"Error decoding image from {image_url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing image from {image_url}: {e}")
        return None

def recognize_faces(face_image, students, tolerance=0.6):
    """Recognizes faces in the input image using student data."""
    for student in students:
        if student.get("images"):
            student_base64_images = []
            for image_url in student["images"]:
                base64_image = download_and_encode_image(image_url)
                if base64_image:
                    student_base64_images.append(base64_image)
            student['base64_images'] = student_base64_images
            del student['images']

    face_encodings = face_recognition.face_encodings(face_image)
    if not face_encodings:
        return None
    face_encoding = face_encodings[0]

    for student in students:
        if student.get("base64_images"):
            for image_base64 in student["base64_images"]:
                try:
                    image_data = cv2.imdecode(np.frombuffer(base64.b64decode(image_base64), np.uint8), cv2.IMREAD_COLOR)
                    known_face_encodings = face_recognition.face_encodings(image_data)
                    if known_face_encodings:
                        known_face_encoding = known_face_encodings[0]
                        results = face_recognition.compare_faces([known_face_encoding], face_encoding, tolerance=tolerance)
                        if results[0]:
                            return student
                except Exception as e:
                    print(f"Error processing image: {e}")

    return None