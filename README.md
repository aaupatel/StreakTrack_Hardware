# StreakTrack - Smart Attendance System

## Overview

StreakTrack is a smart attendance system that utilizes facial recognition to automatically mark the attendance of students. Built on a Raspberry Pi, it connects to a web server via WebSockets to receive student data and send attendance records. The system provides real-time feedback through an LCD display and LED indicators.

## Features

* **Facial Recognition:** Employs the `face_recognition` library to identify students from live camera feeds.
* **Real-time Attendance Marking:** Automatically marks attendance upon successful facial recognition.
* **WebSocket Communication:** Connects to a remote web server using WebSockets for data exchange.
* **Student Data Fetching:** Downloads student information (including image URLs) from the server.
* **Local Face Encoding Storage:** Stores face encodings in a local SQLite database for efficient recognition.
* **Live Video Streaming (Optional):** Streams the camera feed to the web server for monitoring.
* **LCD Display Feedback:** Provides immediate feedback to the user, such as "Please Wait...", student names upon recognition, and status messages.
* **LED Indicators:** Uses different colored LEDs to indicate system status (connected, face detected, attendance marked, already marked, unknown person, errors).
* **Configuration via JSON:** Settings like the web server URL and device/organization IDs are managed through a `config.json` file.
* **Attendance Logging:** Saves attendance records locally in `attendance.json` (cleared on each new successful connection).

## Hardware Requirements

* Raspberry Pi (tested with Raspberry Pi 4)
* Raspberry Pi Camera Module V2 (or compatible)
* LCD Display (20x4 I2C)
* Assorted LEDs (Green, Yellow, Blue, Red)
* Jumper wires
* Power supply for Raspberry Pi

## Software Requirements

* Raspberry Pi OS (or compatible)
* Python 3
* Required Python libraries (install using `pip install -r requirements.txt`):
    * `asyncio`
    * `websockets`
    * `aiohttp`
    * `opencv-python` (`cv2`)
    * `RPi.GPIO`
    * `picamera2`
    * `numpy`
    * `face_recognition`
    * `pickle`
    * `sqlite3`
    * `RPLCD`

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aaupatel/StreakTrack_Hardware.git
    cd StreakTrack_Hardware
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure `config.json`:**
    * Create a `config.json` file in the project root directory based on the `config.example.json` (if provided) or the following structure:
    ```json
    {
      "website_url": "YOUR_WEBSOCKET_SERVER_URL",
      "deviceId": "YOUR_DEVICE_ID",
      "organizationId": "YOUR_ORGANIZATION_ID"
    }
    ```
    * Replace the placeholder values with your actual WebSocket server URL, device ID, and organization ID provided by your StreakTrack backend.

4.  **Hardware Setup:**
    * Connect the Raspberry Pi Camera Module to the CSI port.
    * Connect the LCD display to the Raspberry Pi's I2C pins (typically SDA to GPIO2, SCL to GPIO3, VCC to 5V, and GND to GND).
    * Connect the LEDs to the specified GPIO pins (GREEN: 32, YELLOW: 36, BLUE: 38, RED: 40) via suitable current-limiting resistors. Ensure you understand the Raspberry Pi's GPIO pin numbering (using BOARD numbering in the code).

## Usage

1.  **Run the `main.py` script:**
    ```bash
    sudo python3 main.py
    ```
    * Use `sudo` to ensure the script has the necessary permissions to access the camera and GPIO pins.

2.  **System Operation:**
    * Upon startup, the system will attempt to connect to the WebSocket server. A blue LED will light up upon successful connection, and "Connected" will be displayed on the LCD.
    * The system will then request and download student data from the server, encode the faces, and store them locally. "Students received" will be briefly displayed on the LCD.
    * The camera will continuously capture frames, and the system will attempt to recognize faces.
    * **Face Detected:** The yellow LED will briefly blink, and "Please Wait..." will be displayed on the LCD.
    * **Recognized Student:**
        * If the student's face is recognized and their attendance hasn't been marked yet in the current session:
            * The green LED will light up for 2 seconds.
            * The student's name and "Attendance marked" will be displayed on the LCD for 2 seconds.
            * Attendance data will be sent to the server and saved locally in `attendance.json`.
        * If the student's face is recognized and their attendance has already been marked:
            * The yellow LED will blink once.
            * The student's name and "Already marked" will be displayed on the LCD for 2 seconds.
    * **Unknown Person:** The LCD will display "Unknown Person," and the red LED will blink for 1 second.
    * **Live Streaming (if enabled on the server):** The camera feed will be streamed to the web server. The streaming can be toggled via WebSocket messages ("start\_stream" and "stop\_stream").

3.  **Stopping the script:**
    * Press `Ctrl + C` in the terminal to stop the script. This will also trigger the GPIO cleanup.

## File Structure

StreakTrack/
├── config.json           # Configuration file for server URL and device IDs
├── attendance.json       # Local storage for marked attendance (cleared on new connection)
├── student_faces.db      # Local database for storing face encodings
├── main.py               # Main application script
├── face_utils.py         # Utility functions for camera setup and face recognition
├── stream_utils.py       # Utility functions for encoding video frames for streaming
├── utils.py              # General utility functions (JSON handling, LCD control, LED blinking)
├── requirements.txt      # List of Python dependencies
└── README.md             # This file

## Potential Improvements

* More robust error handling and logging.
* Optimization of face recognition performance.
* Enhanced user interface on the LCD.
* More sophisticated LED feedback.
* Handling of multiple faces in a single frame.
* Dynamic adjustment of streaming quality.
* Offline attendance marking with later synchronization.
* Security considerations for WebSocket communication.

## Contributing

Contributions to this project are welcome. Please feel free to submit pull requests or open issues for bug fixes or feature requests.
