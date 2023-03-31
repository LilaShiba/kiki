import io
import picamera
import logging
import socketserver
from threading import Condition
from http import server
from flask import Flask, render_template, Response, request
import os
import time
import cv2

# Set up the Flask app
app = Flask(__name__)

# Set the desktop directory path
desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")

# Set up the Raspberry Pi camera
camera = picamera.PiCamera(resolution='640x480', framerate=24)

# Set up a global buffer for the video frames
video_buffer = io.BytesIO()

# Set up a condition variable for synchronization
frame_ready = Condition()

# Define a Flask route for the video stream
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define a Flask route for the capture button
@app.route('/capture')
def capture():
    # Create a unique filename using the current timestamp
    filename = time.strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"

    # Convert the latest video frame to an image
    video_buffer.seek(0)
    frame = cv2.imdecode(np.frombuffer(video_buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    # Save the image to the desktop directory
    cv2.imwrite(os.path.join(desktop_dir, filename), frame)

    # Return a response indicating success
    return "Capture successful: " + filename

# Define a generator function to capture video frames and stream them to the Flask app
def generate():
    global video_buffer
    global frame_ready

    while True:
        with frame_ready:
            # Wait for a new video frame to be available
            frame_ready.wait()

            # Copy the latest video frame to the buffer
            camera.capture(video_buffer, format='jpeg', use_video_port=True)
            video_buffer.seek(0)

        # Yield the latest video frame as a byte string
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + video_buffer.getvalue() + b'\r\n')

# Define a function to run the Flask app and video stream server
def run():
    try:
        # Start the video capture and streaming
        camera.start_preview()
        time.sleep(2)
        app.run(host='0.0.0.0', threaded=True)
    finally:
        # Clean up resources
        camera.stop_preview()
        camera.close()

if __name__ == '__main__':
    run()
