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
camRgb = pipeline.create (dai.node.ColorCamera)
#xoutRgb = pipeline.create (dai.node.XLinkOut)
#xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(WIDTH, HEIGHT)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder. RGB)
# Linking
#camRgb.preview.link ( xoutRgb.input )
videoQueue = camRgb.video.createOutputQueue()

# Connect to device and start pipeline
pipeline.start()
while True :
    inRgb = qRgb.get ()
    img = inRgb.getCvFrame()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray ( img )
    # Resize the image
    im_pil = im_pil.resize (( WIDTH , HEIGHT ) )
    # display image on lcd
    disp.display (im_pil)
    if cv2.waitKey ( 1 ) == ord ('q') :
        break
    