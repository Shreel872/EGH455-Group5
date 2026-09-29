import sys
import numpy as py
import cv2
from PIL import Image
import st7735 
import depthai as dai
from PIL import Image , ImageDraw , ImageFont
from fonts .ttf import RobotoMedium as UserFont
import logging
import subprocess

# take the LCD display
subprocess.run(["sudo", "systemctl", "stop", "ip-display"], check=False)

# Create ST7735 LCD display class .
# Create LCD class instance .
disp = st7735.ST7735 (
port = 0,
cs = 1,
dc = 9,
backlight = 12,
rotation = 270,
spi_speed_hz = 10000000
)

# Initalize display
disp.begin ()
WIDTH = disp.width
HEIGHT = disp.height

pipeline = dai.Pipeline ()

# Define source and output
camRgb = pipeline.create (dai.node.Camera).build()

# Properties
cameraOutput = camRgb.requestOutput((WIDTH, HEIGHT), type=dai.ImgFrame.Type.RGB888p)
outputQueue = cameraOutput.createOutputQueue()


# Connect to device and start pipeline

pipeline.start()
while pipeline.isRunning() :
    videoIn = outputQueue.get ()
    img = videoIn.getCvFrame()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray ( img )
    # Resize the image
    im_pil = im_pil.resize (( WIDTH , HEIGHT ) )
    disp.display (im_pil)

    