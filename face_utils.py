import cv2
import face_recognition

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def recognize_faces(face_image, students):
    face_encodings = face_recognition.face_encodings(face_image)
    if not face_encodings:
        return None
    face_encoding = face_encodings[0]

    for student in students:
        if student.get("images"):
            for image_base64 in student["images"]:
                try:
                    image_data = cv2.imdecode(np.frombuffer(base64.b64decode(image_base64), np.uint8), cv2.IMREAD_COLOR)
                    known_face_encodings = face_recognition.face_encodings(image_data)
                    if known_face_encodings:
                        known_face_encoding = known_face_encodings[0]
                        results = face_recognition.compare_faces([known_face_encoding], face_encoding)
                        if results[0]:
                            return student
                except Exception as e:
                    print(f"Error processing image: {e}")

    return None