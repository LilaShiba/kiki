import RPi.GPIO as GPIO
import time

SWITCH_PIN = 16

GPIO.setmode(GPIO.BCM)# Set GPIO pin 26 as input with a pull-up resistor
GPIO.setup(SWITCH_PIN, GPIO.IN)

# Set initial state
state = GPIO.input(SWITCH_PIN)

while True:
    # Read input state
    input_state = GPIO.input(SWITCH_PIN)

    # If input state has changed
    if input_state != state:
        state = input_state

        # If switch is turned on
        if state == GPIO.HIGH:
            print("Switch turned ON")
            # run your program here
            # example: os.system("sudo python3 /home/pi/my_program.py")

    # Wait a short time before checking again
    time.sleep(0.1)
