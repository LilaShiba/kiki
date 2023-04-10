import RPi.GPIO as GPIO
import subprocess
import time

SWITCH_PIN = 26
# Set up the GPIO pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(SWITCH_PIN, GPIO.IN)
# Define the function to run the program
def run_program():
    #subprocess.call(["/path/to/your/program"])
    print('me-wow!')

# Wait for the switch to be pressed
continous = 0
while True:
    # read the state of the channel
    input_state = GPIO.input(SWITCH_PIN)
    # perform some action based on the input state
    if input_state == GPIO.LOW:
        flag = True 
    else:
        flag = False
    
    continous += 1
    if flag:
        continous = 0

    if continous > 150:
        print('meow woof woof')
        

                              