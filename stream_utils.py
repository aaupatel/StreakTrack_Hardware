import cv2
import base64

def encode_frame(frame):
    _, frame_encoded = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(frame_encoded).decode('utf-8')
    return frame_base64