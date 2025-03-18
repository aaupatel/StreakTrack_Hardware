import cv2
import face_recognition
import numpy as np
import json

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def recognize_faces(face_image, students, tolerance=0.6):
    """Recognizes faces in the input image using student data."""
    face_encodings = face_recognition.face_encodings(face_image)
    if not face_encodings:
        return None
    face_encoding = face_encodings[0]

    for student in students:
        if student.get("face_encoding") is not None:
            results = face_recognition.compare_faces([student["face_encoding"]], face_encoding, tolerance=tolerance)
            if results[0]:
                return student
        else:
            try:
                images = []
                for image_url in student.get("images", []):
                    try:
                        resp = requests.get(image_url)
                        img_array = np.frombuffer(resp.content, np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is not None:
                            images.append(img)
                    except Exception as e:
                        print(f"Error loading image from {image_url}: {e}")

                if images:
                    encodings = []
                    for img in images:
                        enc = face_recognition.face_encodings(img)
                        if len(enc) > 0:
                            encodings.append(enc[0])
                    if encodings:
                        student["face_encoding"] = np.mean(encodings, axis=0) #use mean to create a single encoding.
                        results = face_recognition.compare_faces([student["face_encoding"]], face_encoding, tolerance=tolerance)
                        if results[0]:
                            return student
                else:
                    print(f"No valid images found for {student['name']}")

            except Exception as e:
                print(f"Error processing images for {student['name']}: {e}")

    return None

# Load student data
try:
    with open("student_data.json", "r") as f:
        students = json.load(f)
except FileNotFoundError:
    print("Error: student_data.json not found.")
    students = []

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    faces = detect_faces(frame)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Attempt face recognition
        recognized_student = recognize_faces(frame[y:y+h, x:x+w], students)
        if recognized_student:
            print(f"Recognized: {recognized_student['name']}")
            cv2.putText(frame, recognized_student['name'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Detection and Recognition', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()