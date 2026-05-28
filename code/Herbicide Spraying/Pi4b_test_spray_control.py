import RPi.GPIO as GPIO
import time

PIN = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN, GPIO.OUT)

while True:
    GPIO.output(PIN, GPIO.HIGH)
    print("Spray ON")
    time.sleep(2)

    