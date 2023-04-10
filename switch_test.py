import RPi.GPIO as GPIO
import subprocess

# Set up the GPIO pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Define the function to run the program
def run_program():
    subprocess.call(["/path/to/your/program"])

# Wait for the switch to be pressed
while True:
    if GPIO.input(18) == GPIO.LOW:
        run_program()

