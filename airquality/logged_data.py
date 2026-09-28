#!/usr/bin/env python3

import logging
import time

from enviroplus import gas # for gas readings
from bme280 import BME280 #for temperature
from smbus2 import SMBus 

try:
    # Transitional fix for breaking change in LTR559
    from ltr559 import LTR559
    ltr559 = LTR559()
except ImportError:
    import ltr559


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S")

logging.info("""light.py - Print readings from the LTR559 Light & Proximity sensor.

Press Ctrl+C to exit!

""")

# configure weather readings
bus = SMBus(1)
bme280 = BME280(i2c_dev=bus)

# the temperature initally read is affected by the CPU temperature
# so, we must compensate for it
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

try:
    while True:
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

        logging.info(f"""Light: {latest_light:05.02f} Temp: {latest_temperature:05.02f} 
        Pressure: {latest_pressure:05.02f} Humidity: {latest_humidity:05.02f} 
        Oxidised: {latest_oxidised:05.02f} reduced: {latest_reduced:05.02f} 
        nh3: {latest_nh3:05.02f}

""")
        time.sleep(1.0)
except KeyboardInterrupt:
    pass
