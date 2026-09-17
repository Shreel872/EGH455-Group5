#!/usr/bin/env python3

import colorsys
import os
import sys
import time

import st7735

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
        draw.rectangle((i, top_pos, i + 1, HEIGHT), (r, g, b))
        # Draw a line graph in black
        line_y = HEIGHT - (top_pos + (colours[i] * (HEIGHT - top_pos))) + top_pos
        draw.rectangle((i, line_y, i + 1, line_y + 1), (0, 0, 0))
    # Write the text at the top in black
    draw.text((0, 0), message, font=font, fill=(0, 0, 0))
    st7735.display(img)


# Get the temperature of the CPU for compensation
def get_cpu_temperature():
    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
        temp = f.read()
        temp = int(temp) / 1000.0
    return temp


# Tuning factor for compensation. Decrease this number to adjust the
# temperature down, and increase to adjust up
factor = 2.25

cpu_temps = [get_cpu_temperature()] * 5

delay = 0.5  # Debounce the proximity tap
mode = 0  # The starting mode
last_page = 0
light = 1

# Create a values dict to store the data
variables = ["temperature",
             "pressure",
             "humidity",
             "light",
             "oxidised",
             "reduced",
             "nh3"]

values = {}

for v in variables:
    values[v] = [1] * WIDTH

# The main loop
try:
    while True:
        # these variable should always be recorded
        proximity = ltr559.get_proximity()
        
        latest_light = ltr559.get_lux() # in Lux
        latest_pressure = bme280.get_pressure() # in hPa
        latest_humidity = bme280.get_pressure() # in %
        gas_data = gas.read_all() # the gas readings do not give an exact concentration. 
        # these readings are given in ohms, which provides an overview of their concentration
        # as concentration increases, the resistance value will drop
        # you need to run for 10 minutes to set a baseline, and from there you can properly 
        # consider air quality changes. see https://learn.pimoroni.com/article/getting-started-with-enviro-plus
        latest_oxidised = gas_data.oxidising / 1000 # in Ohms
        latest_reduced = gas_data.reducing / 1000 # in Ohms
        latest_nh3 = gas_data.nh3 / 1000 # in Ohms

        # the compensated temperature 
        cpu_temp = get_cpu_temperature()
        # Smooth out with some averaging to decrease jitter
        cpu_temps = cpu_temps[1:] + [cpu_temp]
        avg_cpu_temp = sum(cpu_temps) / float(len(cpu_temps))
        raw_temp = bme280.get_temperature()
        latest_temperature = raw_temp - ((avg_cpu_temp - raw_temp) / factor) # in °C

        # data needs to be sent to the web server; to do to the values need to be logged
        logging.info(f"""Light: {latest_light:05.02f} Temp: {latest_temperature:05.02f} 
        Pressure: {latest_pressure:05.02f} Humidity: {latest_humidity:05.02f} 
        Oxidised: {latest_oxidised:05.02f} Reduced: {latest_reduced:05.02f} 
        NH3: {latest_nh3:05.02f} """)
        time.sleep(1.0)

        # If the proximity crosses the threshold, toggle the mode
        if proximity > 1500 and time.time() - last_page > delay:
            mode += 1
            mode %= len(variables)
            last_page = time.time()

        # One mode for each display variable (IP, video, temperature)
        if mode == 0:
            # IP Address
            ## placeholder
            unit = "%"
            data = bme280.get_humidity()
            display_text(variables[mode], data, unit)


        if mode == 1:
            # video feed
            ## placeholder
            unit = "%"
            data = bme280.get_pressure()
            display_text(variables[mode], data, unit)


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
    sys.exit(0)
