from picamera2 import Picamera2, Preview
import time

def live_preview():
    """Displays a live camera preview."""
    try:
        picam2 = Picamera2()
        #use Preview.QT for desktop environment, and Preview.NULL for headless.
        picam2.start_preview(Preview.QT) # Or Preview.X if you are using X11

        config = picam2.create_preview_configuration(main={"size": (640, 480)}) #adjust size as needed
        picam2.configure(config)

        picam2.start()

        print("Live preview started. Press Ctrl+C to stop.")

        while True:
            time.sleep(0.1) # Add a small delay to reduce CPU usage.

    except KeyboardInterrupt:
        print("\nLive preview stopped.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if 'picam2' in locals() and picam2 is not None:
            picam2.stop()

if __name__ == "__main__":
    live_preview()