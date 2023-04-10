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
    # read the state of the channel
    input_state = GPIO.input(SWITCH_PIN)

    # perform some action based on the input state
    if input_state == GPIO.HIGH:
        print("Button is not pressed")
    else:
        print("Button is pressed")

