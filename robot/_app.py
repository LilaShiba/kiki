import RPi.GPIO as GPIO
from flask import Flask, render_template, Response
import subprocess
import picamera
import io
import time
import cv2
import os

app = Flask(__name__)

# TODO Setup fan controls
img_cnt = 0
# PIR MOTION SENSOR
PIR_PIN = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)


# Servo 1 -> orientation
# servo_pin = 18 # GPIO pin connected to servo signal wire
# freq = 50 # PWM frequency in Hz
# duty_min = 2.5 # duty cycle for minimum servo position in percent
# duty_max = 12.5 # duty cycle for maximum servo position in percent
# GPIO.setup(servo_pin, GPIO.OUT)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pir_data')
def pir_data():
    return Response(generate_pir_data(), mimetype='text/event-stream')

@app.route('/video_feed')
def video_feed():
    
    return Response(gen() , mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sensor_data')
def sensor_data():
    # Wait for the sensor to settle
    print("Waiting for sensor to settle...")
    time.sleep(2)
    print("Ready")

    try:
        while True:
            # Read the sensor value
            pir_value = GPIO.input(PIR_PIN)

            # Print the sensor value
            #if pir_value:
                #print("Motion detected!")
            

            # Wait for a short time before reading the sensor again
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Exiting program...")
    finally:
        # Clean up the GPIO pins
        GPIO.cleanup()

# TODO pipe jiji through here
@app.route('/run_jiji',methods=['POST'])
def run_jiji():
    get_current_img()
    script_output = subprocess.check_output(['python', 'scripts/test.py'])
    return render_template('result.html', output=script_output)

# Helpers
def gen():
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.framerate = 30
        camera.start_preview()
        while True:
            frame = get_frame(camera)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        

def generate_pir_data():
    while True:
        pir_state = GPIO.input(PIR_PIN)
        yield f"data: {pir_state}\n\n"
        time.sleep(0.1)

def get_frame(camera):
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)