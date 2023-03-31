import os
import io
import time
import picamera
import subprocess
import numpy as np
import RPi.GPIO as GPIO
from flask import Flask, render_template, Response


app = Flask(__name__)

# TODO Setup fan controls
img_cnt = 0
# PIR MOTION SENSOR
PIR_PIN = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
camera.brightness = 60
camera.contrast = 30
camera.start_preview()
# Global camera crashes GPU
    
# Helpers

def gen():
    while True:
        frame = get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
def generate_pir_data():
    while True:
        pir_state = GPIO.input(PIR_PIN)
        yield f"data: {pir_state}\n\n"
        time.sleep(0.1)

def get_frame():
    stream = io.BytesIO()
    camera.capture(stream, format='jpeg', use_video_port=True)
    frame = stream.getvalue()
    stream.seek(0)
    stream.truncate()
    return frame

def set_servo_pos(pos):
    # setup PWM
    pwm = GPIO.PWM(servo_pin, freq)
    pwm.start(0)
    duty = duty_min + (pos/180)*(duty_max - duty_min)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.3) # wait for servo to reach position

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the video stream
@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the capture button
@app.route('/capture')
def capture():
    # Capture a frame from the video stream
    img_file_name = "imgs/"+time.strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
    img = get_frame(camera)
    camera.capture(img_file_name)
    # Display the processed image on a separate page
    return render_template('capture.html', filename=img_file_name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
