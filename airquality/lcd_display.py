#!/usr/bin/env python3

import colorsys
import os
import sys
import time
import signal
import socket


import numpy as py
import cv2
from PIL import Image
import st7735 
import depthai as dai
from PIL import Image , ImageDraw , ImageFont
from fonts .ttf import RobotoMedium as UserFont
import logging

import subprocess

import st7735

subprocess.run(["sudo", "systemctl", "stop", "ip-display"], check=False)

from fonts.ttf import RobotoMedium as UserFont
from PIL import Image, ImageDraw, ImageFont

POLL_SECONDS = 5

try:
    # Transitional fix for breaking change in LTR559
    from ltr559 import LTR559
    ltr559 = LTR559()
except ImportError:
    import ltr559

import logging

from bme280 import BME280
from fonts.ttf import RobotoMedium as UserFont
from PIL import Image, ImageDraw, ImageFont

from enviroplus import gas

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S")

logging.info("""all-in-one.py - Displays readings from all of Enviro plus' sensors
Press Ctrl+C to exit!
""")

# BME280 temperature/pressure/humidity sensor
bme280 = BME280()

# Create ST7735 LCD display class
st7735 = st7735.ST7735(
    port=0,
    cs=1,
    dc="GPIO9",
    backlight="GPIO12",
    rotation=270,
    spi_speed_hz=10000000
)

# Initialize display
st7735.begin()

WIDTH = st7735.width
HEIGHT = st7735.height

# Set up canvas and font
img = Image.new("RGB", (WIDTH, HEIGHT), color=(0, 0, 0))
draw = ImageDraw.Draw(img)
path = os.path.dirname(os.path.realpath(__file__))
font_size = 20
font = ImageFont.truetype(UserFont, font_size)

message = ""

# The position of the top bar
top_pos = 25



# Displays data and text on the 0.96" LCD
def display_text(variable, data, unit):
    local_img = Image.new("RGB", (WIDTH, HEIGHT), color = (255, 255, 255))
    local_draw = ImageDraw.Draw(local_img)
    # Maintain length of list
    values[variable] = values[variable][1:] + [data]
    # Scale the values for the variable between 0 and 1
    vmin = min(values[variable])
    vmax = max(values[variable])
    colours = [(v - vmin + 1) / (vmax - vmin + 1) for v in values[variable]]
    # Format the variable name and value
    message = f"{variable[:4]}: {data:.1f} {unit}"
    logging.info(message)
    draw.rectangle((0, 0, WIDTH, HEIGHT), (255, 255, 255))
    for i in range(len(colours)):
        # Convert the values to colours from red to blue
        colour = (1.0 - colours[i]) * 0.6
        r, g, b = [int(x * 255.0) for x in colorsys.hsv_to_rgb(colour, 1.0, 1.0)]
        # Draw a 1-pixel wide rectangle of colour
        local_draw.rectangle((i, top_pos, i + 1, HEIGHT), (r, g, b))
        # Draw a line graph in black
        line_y = HEIGHT - (top_pos + (colours[i] * (HEIGHT - top_pos))) + top_pos
        local_draw.rectangle((i, line_y, i + 1, line_y + 1), (0, 0, 0))
    # Write the text at the top in black
    local_draw.text((0, 0), message, font=font, fill=(0, 0, 0))
    st7735.display(local_img)


# Get the temperature of the CPU for compensation
def get_cpu_temperature():
    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
        temp = f.read()
        temp = int(temp) / 1000.0
    return temp

# also get the IP address of the Pi
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.5)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()

# and set up how the IP will be drawn
def centre(draw, text, font, y):
    x1, _, x2, _ = font.getbbox(text)
    draw.text(((WIDTH - (x2 - x1)) / 2, y), text, font=font, fill=TEXT)


def render(ip):
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, WIDTH, HEIGHT), BACK if ip else BACK_DOWN)
    #centre(draw, socket.gethostname(), font_small, 8)
    centre(draw, ip or "no network", font_big, 32)
    #centre(draw, time.strftime("%H:%M"), font_small, 60)
    st7735.display(img)


def shutdown(signum, frame):
    st7735.set_backlight(0)
    subprocess.run(["sudo","-n","systemctl", "start", "ip-display"], check=False)
    sys.exit(0)

font_big = ImageFont.truetype(UserFont, 18)
font_small = ImageFont.truetype(UserFont, 13)

BACK = (0, 90, 130)
BACK_DOWN = (130, 60, 0)
TEXT = (255, 255, 255)


# Tuning factor for compensation. Decrease this number to adjust the
# temperature down, and increase to adjust up
factor = 2.25

cpu_temps = [get_cpu_temperature()] * 5

# set up the video stream
pipeline = dai.Pipeline ()

# Define source and output
camRgb = pipeline.create (dai.node.Camera).build()

# Properties
cameraOutput = camRgb.requestOutput((WIDTH, HEIGHT), type=dai.ImgFrame.Type.RGB888p)
outputQueue = cameraOutput.createOutputQueue()

# Connect to device and start pipeline
pipeline.start()

delay = 0.5  # Debounce the proximity tap
mode = 0  # The starting mode
last_page = 0
light = 1

# Create a values dict to store the data
variables = ["ip_address",
             "video_feed",
             "temperature"]

values = {}

for v in variables:
    values[v] = [1] * WIDTH

# The main loop
try:
    while True:
        # these variable should always be recorded
        proximity = ltr559.get_proximity()

        # If the proximity crosses the threshold, toggle the mode
        if proximity > 1500 and time.time() - last_page > delay:
            mode += 1
            mode %= len(variables)
            last_page = time.time()

        # One mode for each display variable (IP, video, temperature)
        if mode == 0:
            # IP Address
            # systemd sends SIGTERM on stop -- blank the screen rather than freezing it
            signal.signal(signal.SIGTERM, shutdown)
            signal.signal(signal.SIGINT, shutdown)

            last = object()  # sentinel so the first pass always draws

            ip = get_ip()
            if ip != last:
                render(ip)
                last = ip



        if mode == 1:
            # video feed
            videoIn = outputQueue.get ()
            cam_img = videoIn.getCvFrame()
            cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray ( cam_img )
            # Resize the image
            im_pil = im_pil.resize (( WIDTH , HEIGHT ) )
            st7735.display (im_pil)
            


        if mode == 2:
        # variable = "temperature"
            unit = "°C"
            cpu_temp = get_cpu_temperature()
            # Smooth out with some averaging to decrease jitter
            cpu_temps = cpu_temps[1:] + [cpu_temp]
            avg_cpu_temp = sum(cpu_temps) / float(len(cpu_temps))
            raw_temp = bme280.get_temperature()
            data = raw_temp - ((avg_cpu_temp - raw_temp) / factor)
            display_text(variables[mode], data, unit)

# Exit cleanly
except KeyboardInterrupt:
    subprocess.run(["sudo","-n","systemctl", "start", "ip-display"], check=False)
    sys.exit(0)