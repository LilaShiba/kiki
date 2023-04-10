import RPi.GPIO as GPIO
import subprocess
import time

SWITCH_PIN = 26
# Set up the GPIO pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(SWITCH_PIN, GPIO.IN)
continous = 0
# Define the function to run the program
def run_program():
    #subprocess.call(["/path/to/your/program"])
    print('me-wow!')

# Wait for the switch to be pressed

while True:
    # read the state of the channel
    input_state = GPIO.input(SWITCH_PIN)
    # perform some action based on the input state
    if input_state == GPIO.LOW:
        flag = False
    else:
        flag = True
    continous += 1
    if flag:
        continous = 0
    if continous > 50:
         print("Button is pressed")

                              