import cv2
import face_recognition
import numpy as np
import base64
import requests
import os

def detect_faces(frame):
    face_locations = face_recognition.face_locations(frame)
    faces = [(left, top, (right - left), (bottom - top)) for top, right, bottom, left in face_locations]
    return faces
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # return faces

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
                image_filename = image_url.split('/')[-1]
                cache_path = f"cache_images/{image_filename}"
                if os.path.exists(cache_path):
                    with open(cache_path, "rb") as f:
                        image_base64 = base64.b64encode(f.read()).decode('utf-8')
                else:
                    base64_image = download_and_encode_image(image_url)
                    if base64_image:
                        image_data = base64.b64decode(base64_image)
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        with open(cache_path, "wb") as f:
                            f.write(image_data)
                        image_base64 = base64_image

            if image_base64:
                student_base64_images.append(image_base64)
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