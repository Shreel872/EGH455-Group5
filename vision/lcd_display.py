#!/usr/bin/env python3
"""Show OAK ArUco detections on an Enviro+ ST7735 LCD.

The camera and ArUco detector run in one worker thread. Only the main thread
writes to the LCD. A proximity tap switches between marker, IP and temperature.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
import logging
import signal
import socket
import subprocess
import threading
import time

import cv2
import depthai as dai
from PIL import Image, ImageDraw, ImageFont
import st7735
from bme280 import BME280


LOG = logging.getLogger(__name__)
PAGES = ("marker", "ip", "temperature")


@dataclass(frozen=True)
class Detection:
    ids: tuple[int, ...] = ()
    dictionary: str = ""
    status: str = "STARTING CAMERA"
    observed_at: float = 0.0
    rejected: int = 0


class SharedDetection:
    def __init__(self):
        self._lock = threading.Lock()
        self._value = Detection()

    def put(self, ids=(), dictionary="", status="NO MARKER", rejected=0):
        value = Detection(tuple(ids), dictionary, status, time.monotonic(), rejected)
        with self._lock:
            self._value = value

    def get(self, stale_seconds):
        with self._lock:
            value = self._value
        if value.status not in ("CAMERA ERROR", "CAMERA STOPPED") and value.observed_at:
            if time.monotonic() - value.observed_at > stale_seconds:
                return Detection(status="CAMERA STALE", dictionary=value.dictionary)
        return value


def make_detector(name):
    key = "DICT_" + name.upper().removeprefix("DICT_")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
        raise RuntimeError("OpenCV ArUco is unavailable; install opencv-contrib-python")
    if not hasattr(cv2.aruco, key):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, key))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


class DictionarySearch:
    """Probe families one at a time; confirm a match on the following frame."""

    def __init__(self, requested):
        self.auto = requested.lower() == "auto"
        if self.auto:
            names = [f"{bits}X{bits}_{size}" for bits in (4, 5, 6, 7)
                     for size in (50, 100, 250, 1000)]
            names.append("ARUCO_ORIGINAL")
        else:
            names = [requested.upper().removeprefix("DICT_")]
        self.detectors = [(name, make_detector(name)) for name in names]
        self.next_index = 0
        self.candidate = None
        self.locked = None if self.auto else 0
        self.misses = 0

    def detect(self, gray):
        index = (self.locked if self.locked is not None else
                 self.candidate[0] if self.candidate is not None else self.next_index)
        name, detector = self.detectors[index]
        _, ids, rejected = detector.detectMarkers(gray)
        found = () if ids is None else tuple(sorted(int(value) for value in ids.flatten()))

        if self.locked is not None:
            self.misses = 0 if found else self.misses + 1
            if self.auto and self.misses >= 30:
                self.locked = None
                self.candidate = None
                self.next_index = (index + 1) % len(self.detectors)
                self.misses = 0
            return found, name, len(rejected), self.locked is not None

        if self.candidate is not None and self.candidate == (index, found) and found:
            self.locked = index
            self.candidate = None
            LOG.info("Locked ArUco dictionary: %s", name)
            return found, name, len(rejected), True

        if found:
            self.candidate = (index, found)
        else:
            self.candidate = None
            self.next_index = (index + 1) % len(self.detectors)
        return (), name, len(rejected), False


def camera_worker(args, shared, stop):
    try:
        search = DictionarySearch(args.dictionary)
        with dai.Pipeline() as pipeline:
            camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            output = camera.requestOutput(
                (args.width, args.height),
                dai.ImgFrame.Type.BGR888p,
                dai.ImgResizeMode.LETTERBOX,
                args.fps,
            )
            queue = output.createOutputQueue(maxSize=1, blocking=False)
            pipeline.start()
            LOG.info("OAK camera started at %sx%s, %.1f FPS", args.width, args.height, args.fps)
            last_logged = None
            while not stop.is_set() and pipeline.isRunning():
                packet = queue.tryGet()
                if packet is None:
                    stop.wait(0.02)
                    continue
                frame = packet.getCvFrame()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ids, dictionary, rejected, locked = search.detect(gray)
                status = "DETECTED" if ids else "NO MARKER" if locked else "SEARCHING"
                shared.put(ids, dictionary, status, rejected)
                report = (ids, dictionary if locked else None)
                if report != last_logged:
                    LOG.info("%s; IDs: %s; dictionary: %s; rejected: %s",
                             status, ids or "none", dictionary, rejected)
                    last_logged = report
        if not stop.is_set():
            shared.put(status="CAMERA STOPPED")
    except Exception:
        LOG.exception("Camera or ArUco detection failed")
        shared.put(status="CAMERA ERROR")


def load_font(size):
    try:
        from fonts.ttf import RobotoMedium
        return ImageFont.truetype(RobotoMedium, size)
    except (ImportError, OSError):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def centered(draw, text, font, y, width, color):
    left, top, right, _ = draw.textbbox((0, 0), text, font=font)
    draw.text(((width - (right - left)) / 2 - left, y - top), text,
              font=font, fill=color)


def marker_image(size, fonts, value, now):
    width, height = size
    image = Image.new("RGB", size, (6, 16, 25))
    draw = ImageDraw.Draw(image)
    small, medium, large = fonts
    centered(draw, "ARUCO", small, 5, width, (180, 205, 220))
    detected = value.status == "DETECTED" and bool(value.ids)
    color = (36, 220, 110) if detected else (255, 175, 55)
    centered(draw, value.status, medium, 25, width, color)
    if detected:
        index = int(now / 2) % len(value.ids)
        centered(draw, f"ID {value.ids[index]}", large, 47, width, (255, 255, 255))
        if len(value.ids) > 1:
            draw.text((width - 27, height - 11), f"{index + 1}/{len(value.ids)}",
                      font=small, fill=(180, 205, 220))
    elif value.status == "SEARCHING":
        centered(draw, "Show marker", small, 52, width, (180, 205, 220))
    return image


def info_image(size, fonts, title, line, color):
    width, _ = size
    image = Image.new("RGB", size, (6, 16, 25))
    draw = ImageDraw.Draw(image)
    small, medium, _ = fonts
    centered(draw, title, small, 8, width, (180, 205, 220))
    centered(draw, line, medium, 37, width, color)
    return image


def get_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0.2)
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "NO NETWORK"
def display_text(size, variable, data, unit, values, font):
    width, height = size
    history = values.setdefault(variable, [])
    history.append(data)
    del history[:-width]

    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((2, 2), f"{variable[:4]}: {data:.1f} {unit}",
              font=font, fill=(0, 0, 0))

    low, high = min(history), max(history)
    span = max(high - low, 0.5)
    low = (low + high - span) / 2
    points = [
        (i, height - 2 - int((value - low) / span * (height - 27)))
        for i, value in enumerate(history)
    ]
    if len(points) > 1:
        draw.line(points, fill=(0, 0, 0), width=1)
    else:
        draw.point(points[0], fill=(0, 0, 0))
    return image
def load_sensors():
    try:
        from ltr559 import LTR559
        proximity = LTR559()
    except ImportError:
        try:
            import ltr559 as proximity
        except ImportError:
            proximity = None
    try:
        from bme280 import BME280
        temperature = BME280()
    except (ImportError, OSError):
        temperature = None
    return proximity, temperature


def temperature_text(sensor):
    if sensor is None:
        return "NO SENSOR"
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", encoding="ascii") as handle:
            cpu = int(handle.read()) / 1000.0
        raw = sensor.get_temperature()
        adjusted = raw - ((cpu - raw) / 2.25)
        return f"{adjusted:.1f} C"
    except (OSError, ValueError):
        return "SENSOR ERROR"
def get_cpu_temperature():
    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
        temp = f.read()
        temp = int(temp) / 1000.0
    return temp

def manage_ip_display(action):
    try:
        result = subprocess.run(["sudo", "-n", "systemctl", action, "ip-display"],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            LOG.warning("Could not %s ip-display: %s", action, result.stderr.strip())
        return result.returncode == 0
    except FileNotFoundError:
        LOG.warning("sudo/systemctl is unavailable; ip-display was not changed")
        return False


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", default="auto", help="e.g. 4X4_50, or auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--refresh-hz", type=float, default=5.0)
    parser.add_argument("--stale-seconds", type=float, default=2.0)
    parser.add_argument("--keep-ip-display", action="store_true",
                        help="Do not stop/restart the existing ip-display service")
    args = parser.parse_args()
    if min(args.width, args.height) <= 0 or args.fps <= 0 or args.refresh_hz <= 0:
        parser.error("width, height, fps and refresh-hz must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    restart_service = False
    if not args.keep_ip_display:
        restart_service = manage_ip_display("stop")
    stop = threading.Event()
    lcd = None
    worker = None
    bme280 = BME280()


# The main loop
    try:
        lcd = st7735.ST7735(port=0, cs=1, dc="GPIO9", backlight="GPIO12",
                            rotation=270, spi_speed_hz=10000000)

        lcd.begin()
        size = (lcd.width, lcd.height)
        fonts = (load_font(11), load_font(17), load_font(23))
        proximity, temperature = load_sensors()
        shared = SharedDetection()
        worker = threading.Thread(target=camera_worker, args=(args, shared, stop),
                                  name="aruco-camera", daemon=True)
        worker.start()

        def request_stop(_signum, _frame):
            stop.set()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        page = 0  # Marker status is visible immediately.
        was_near = False
        last_tap = 0.0
        last_ip = ""
        last_ip_at = 0.0
        last_temp = ""
        last_temp_at = 0.0
        factor = 2.25
        cpu_temps = [get_cpu_temperature()] * 5
        values = {"temperature": []}
        LOG.info("LCD ready. Proximity tap cycles marker / IP / temperature.")
        while not stop.is_set():
            now = time.monotonic()
            if proximity is not None:
                try:
                    near = proximity.get_proximity() > 1500
                    if near and not was_near and now - last_tap > 0.7:
                        page = (page + 1) % len(PAGES)
                        last_tap = now
                        LOG.info("LCD page: %s", PAGES[page])
                    was_near = near
                except (OSError, ValueError):
                    LOG.warning("Proximity sensor unavailable; staying on current page")
                    proximity = None

            if PAGES[page] == "marker":
                image = marker_image(size, fonts, shared.get(args.stale_seconds), now)
            elif PAGES[page] == "ip":
                if now - last_ip_at >= 5 or not last_ip:
                    last_ip = get_ip()
                    last_ip_at = now
                image = info_image(size, fonts, "IP ADDRESS", last_ip, (255, 255, 255))
            else:
                if temperature is None:
                    image = info_image(size, fonts, "TEMPERATURE", "NO SENSOR", (255, 255, 255))
                else:
                    cpu_temp = get_cpu_temperature()
                    cpu_temps = cpu_temps[1:] + [cpu_temp]
                    avg_cpu_temp = sum(cpu_temps) / len(cpu_temps)
                    raw_temp = temperature.get_temperature()
                    data = raw_temp - ((avg_cpu_temp - raw_temp) / factor)
                    image = display_text(size, "temperature", data, "C", values, fonts[0])
            lcd.display(image)
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=3)
        if lcd is not None:
            lcd.set_backlight(0)
        if restart_service:
            manage_ip_display("start")


if __name__ == "__main__":
    main()
