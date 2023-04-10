import RPi.GPIO as GPIO
import time

# Set GPIO mode to BCM
GPIO.setmode(GPIO.BCM)

# Set GPIO pin 26 as input with a pull-up resistor
GPIO.setup(26, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Set initial state
state = GPIO.input(26)

while True:
    # Read input state
    input_state = GPIO.input(26)

    # If input state has changed
    if input_state != state:
        state = input_state

        # If switch is turned on
        if state == GPIO.LOW:
            print("Switch turned ON")
            # run your program here
            # example: os.system("sudo python3 /home/pi/my_program.py")

    # Wait a short time before checking again
    time.sleep(0.1)
