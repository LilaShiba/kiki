import RPi.GPIO as GPIO
import time
import subprocess

GPIO.setmode(GPIO.BCM)

IR_LED_PIN = 17  # The GPIO pin number where the IR LED is connected
GPIO.setup(IR_LED_PIN, GPIO.OUT)

def send_ir_signal(pulses):
    for pulse in pulses:
        #GPIO.output(IR_LED_PIN, GPIO.HIGH)
        #time.sleep(pulse[0] / 1000000.0)
        GPIO.output(IR_LED_PIN, GPIO.LOW)
        time.sleep(pulse[1] / 1000000.0)

# IR pulses for various off commands
off_signals = {
    'SPEAKERS': [(800, 400), (800, 1600), (800, 400), (800, 1600), (800, 400), (800, 1600)],
    'BLUETOOTH': [(350, 175), (350, 525), (350, 175), (350, 525), (350, 175), (350, 175), (350, 525)],
    'TV': [(9000, 4500), (600, 550), (600, 1700), (600, 550), (600, 1700), (600, 550), (600, 1700), (600, 550), (600, 1700), (600, 550), (600, 550), (600, 550), (600, 550), (600, 550), (600, 550), (600, 1700)],
    'SOUND_BAR': [(9000, 4500), (450, 450), (450, 1300), (450, 450), (450, 1300), (450, 450), (450, 1300), (450, 450), (450, 450), (450, 450), (450, 450), (450, 450), (450, 450), (450, 450), (450, 450), (450, 450), (450, 450), (450, 1300)],
    'AMPLIFIER': [(9000, 4500), (600, 550), (600, 1700), (600, 550), (600, 1700), (600, 550), (600, 1700), (600, 550), (600, 1700), (600, 550), (600, 550), (600, 550), (600, 550), (600, 550), (600, 1700)],
    #'AIR_CONDITIONER': [(3500, 1750), (550, 550), (550, 550), (550, 550), (550, 550), (550, 550), (550, 550), (550, 550), (550, 550), (550, 1650)]
}

while True:
        # Send the IR signal for the selected device
    for device, value in off_signals.items():
        send_ir_signal(off_signals[device])
        time.sleep(1)  # Wait for 1 second before sending another signal
