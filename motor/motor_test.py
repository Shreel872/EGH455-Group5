#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time
import sys
import tty
import termios

# --- HARDWARE SETUP ---
# Recommended pin: BCM GPIO 18 (Pin 12 on the board) 
# Servos run on a standard 50Hz frequency (20ms period)
PWM_PIN = 13
FREQUENCY = 50 
GPIO.setmode(GPIO.BCM)
GPIO.setup(PWM_PIN, GPIO.OUT)

# Initialize PWM on the pin
pwm = GPIO.PWM(PWM_PIN, FREQUENCY)

# --- 360-SERVO SPEED/DIRECTION VALUES (at 50Hz) ---
# 7.5% duty cycle = Stop (1.5ms pulse)
# 5.0% duty cycle = Full speed clockwise (1.0ms pulse)
# 10.0% duty cycle = Full speed counter-clockwise (2.0ms pulse)
MOTOR_STOP = 7.5
MOTOR_CW = 5.0
MOTOR_CCW = 10.0

# Start motor in stopped position
pwm.start(MOTOR_STOP)

def get_keypress():
    """Reads a single keypress from the terminal without needing Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

print("=========================================")
print("  DF15RSMG 360-Servo Manual Drive Test   ")
print("=========================================")
print("Controls:")
print("  [a] - Rotate Counter-Clockwise (CCW)")
print("  [d] - Rotate Clockwise (CW)")
print("  [s] - Stop Motor")
print("  [q] - Quit & Clean up")
print("-----------------------------------------")

try:
    while True:
        char = get_keypress().lower()
        
        if char == 'a':
            print("\rRotating COUNTER-CLOCKWISE...", end="")
            pwm.ChangeDutyCycle(MOTOR_CCW)
        elif char == 'd':
            print("\rRotating CLOCKWISE...", end="")
            pwm.ChangeDutyCycle(MOTOR_CW)
        elif char == 's':
            print("\rStopping motor...", end="")
            pwm.ChangeDutyCycle(MOTOR_STOP)
        elif char == 'q':
            print("\rExiting test interface.")
            break

except KeyboardInterrupt:
    print("\nTest interrupted by keyboard.")

finally:
    # Always clean up GPIO to release the pin safely
    pwm.stop()
    GPIO.cleanup()
    print("GPIO Cleanup complete.")