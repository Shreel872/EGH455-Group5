#!/usr/bin/env python3
import signal
import socket
import sys
import time

import st7735
from fonts.ttf import RobotoMedium as UserFont
from PIL import Image, ImageDraw, ImageFont

POLL_SECONDS = 5

disp = st7735.ST7735(
    port=0,
    cs=1,
    dc="GPIO9",
    backlight="GPIO12",
    rotation=270,
    spi_speed_hz=10000000,
)
disp.begin()

WIDTH, HEIGHT = disp.width, disp.height

font_big = ImageFont.truetype(UserFont, 18)
font_small = ImageFont.truetype(UserFont, 13)

BACK = (0, 90, 130)
BACK_DOWN = (130, 60, 0)
TEXT = (255, 255, 255)


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
    disp.display(img)


def shutdown(signum, frame):
    disp.set_backlight(0)
    sys.exit(0)


# systemd sends SIGTERM on stop -- blank the screen rather than freezing it
signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

last = object()  # sentinel so the first pass always draws
try:
    while True:
        ip = get_ip()
        if ip != last:
            render(ip)
            last = ip
        time.sleep(POLL_SECONDS)
except KeyboardInterrupt:
    shutdown(None, None)

