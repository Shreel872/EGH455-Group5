#!/usr/bin/env python3

import json
import signal
import socket
import sys
import time
import textwrap

import websocket
import st7735

from fonts.ttf import RobotoMedium as UserFont
from PIL import Image, ImageDraw, ImageFont


DJANGO_IP = "1192.168.1.113"

WS_URL = (
    f"ws://{DJANGO_IP}:8000/ws/lcd/"
)



# LCD SETUP


disp = st7735.ST7735(
    port=0,
    cs=1,
    dc="GPIO9",
    backlight="GPIO12",
    rotation=270,
    spi_speed_hz=10000000,
)

disp.begin()


WIDTH = disp.width
HEIGHT = disp.height


font = ImageFont.truetype(
    UserFont,
    16
)


small_font = ImageFont.truetype(
    UserFont,
    13
)


BACKGROUND = (
    0,
    90,
    130
)

TEXT = (
    255,
    255,
    255
)




latest_temperature = None

latest_humidity = None

latest_pressure = None

latest_target = None

latest_target_state = None




def get_ip_address():

    try:

        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        sock.connect(
            ("8.8.8.8", 80)
        )

        ip_address = (
            sock.getsockname()[0]
        )

        sock.close()

        return ip_address

    except Exception:

        return "No Network"


#text display

def display_text(
    text,
    selected_font=None
):

    if selected_font is None:

        selected_font = font


    disp.set_backlight(1)


    image = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        color=BACKGROUND,
    )


    draw = ImageDraw.Draw(
        image
    )


    lines = []


    for paragraph in str(
        text
    ).split("\n"):

        wrapped = textwrap.wrap(
            paragraph,
            width=18,
        )

        if wrapped:

            lines.extend(
                wrapped
            )

        else:

            lines.append(
                ""
            )


    lines = lines[:5]


    line_height = 18


    total_height = (
        len(lines)
        * line_height
    )


    y = (
        HEIGHT
        - total_height
    ) // 2


    for line in lines:

        bbox = draw.textbbox(
            (0, 0),
            line,
            font=selected_font,
        )


        text_width = (
            bbox[2]
            - bbox[0]
        )


        x = (
            WIDTH
            - text_width
        ) // 2


        draw.text(
            (x, y),
            line,
            font=selected_font,
            fill=TEXT,
        )


        y += line_height


    disp.display(
        image
    )


#Display ip

def display_ip():

    ip_address = (
        get_ip_address()
    )


    display_text(
        "IP Address\n"
        + ip_address
    )


#display detected target

def display_target():

    if latest_target is None:

        display_text(
            "Target Detection\n"
            "Waiting for data"
        )

        return


    text = (
        "Target Detected\n"
        + str(latest_target)
    )


    if latest_target_state:

        text += (
            "\n"
            + str(
                latest_target_state
            )
        )


    display_text(
        text
    )


#display temperatuere

def display_temperature():

    if latest_temperature is None:

        display_text(
            "Temperature\n"
            "Sensor data\n"
            "not connected"
        )

        return


    display_text(
        "Temperature\n"
        f"{latest_temperature:.1f} C"
    )


#dispaly sensor values

def display_sensors():

    if (
        latest_temperature is None
        and latest_humidity is None
        and latest_pressure is None
    ):

        display_text(
            "Sensor Summary\n"
            "Waiting for\n"
            "sensor data"
        )

        return


    lines = [
        "Sensors"
    ]


    if (
        latest_temperature
        is not None
    ):

        lines.append(
            f"T: "
            f"{latest_temperature:.1f} C"
        )


    if (
        latest_humidity
        is not None
    ):

        lines.append(
            f"H: "
            f"{latest_humidity:.1f} %"
        )


    if (
        latest_pressure
        is not None
    ):

        lines.append(
            f"P: "
            f"{latest_pressure:.0f}"
        )


    display_text(
        "\n".join(
            lines
        ),
        small_font,
    )



# clear screen


def clear_screen():

    image = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        color=(0, 0, 0),
    )

    disp.display(
        image
    )


#change display mode

def change_mode(mode):

    print(
        "Changing LCD mode to:",
        mode
    )


    if mode == "ip":

        display_ip()


    elif mode == "target":

        display_target()


    elif mode == "temperature":

        display_temperature()


    elif mode == "sensors":

        display_sensors()




def on_open(ws):

   

    # Default display
    display_ip()




def on_message(
    ws,
    message
):

    print(
        "Received:",
        message
    )


    try:

        data = json.loads(
            message
        )

    except json.JSONDecodeError:

        print(
            "Invalid JSON"
        )

        return


    if (
        data.get("type")
        != "lcd_command"
    ):

        return


    command = data.get(
        "command"
    )



    if command == "mode":

        mode = data.get(
            "mode"
        )

        change_mode(
            mode
        )



    elif command == "display":

        text = data.get(
            "text",
            ""
        )

        display_text(
            text
        )




    elif command == "clear":

        clear_screen()




    elif command == "off":

        disp.set_backlight(
            0
        )




    elif command == "on":

        disp.set_backlight(
            1
        )

        display_ip()




def on_error(
    ws,
    error
):

    print(
        "WebSocket error:",
        error
    )












display_ip()




while True:

    print(
        "Connecting to:",
        WS_URL
    )


    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
       
    )


    ws.run_forever()


    print(
        "Retrying connection "
        "in 2 seconds..."
    )


    time.sleep(
        2
    )