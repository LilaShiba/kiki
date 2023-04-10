import RPi.GPIO as GPIO
import subprocess
SWITCH_PIN = 26
# Set up the GPIO pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Define the function to run the program
def run_program():
    print('meow meow meow')
    #subprocess.call(["/path/to/your/program"])

# Wait for the switch to be pressed
while True:
    if GPIO.input(SWITCH_PIN) == GPIO.HIGH:
        run_program()
