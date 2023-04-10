import RPi.GPIO as GPIO
import subprocess

SWITCH_PIN = 16
# Set up the GPIO pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(SWITCH_PIN, GPIO.IN)

# Define the function to run the program
def run_program():
    #subprocess.call(["/path/to/your/program"])
    print('me-wow!')

# Wait for the switch to be pressed
while True:
    if GPIO.input(SWITCH_PIN) == GPIO.LOW:
        pass
    else:
        run_program()

