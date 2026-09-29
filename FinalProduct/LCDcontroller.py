"""Enviro+ LCD display for sensor and vision results."""

from collections import deque
from copy import deepcopy
import math
import threading
import time


class LCDController:
    PAGES = ("detections", "ip", "temperature")

    def __init__(self, enabled=True, stale_seconds=2.0):
        self.enabled = enabled
        self.stale_seconds = stale_seconds

        self._display = None
        self._owner = None

        self._lock = threading.Lock()
        self._latest = {}

        self._page = 0
        self._was_near = False
        self._last_tap = -float("inf")

        self._last_temperature_at = None
        self._history = deque(maxlen=160)

    def update(self, source, data, observed_at=None):
        """Publish readings from any worker thread.

        source: aruco, ip, sensors, valve or gauge.
        observed_at: time.monotonic() timestamp, if supplied.
        """
        if source not in (
            "aruco", "ip", "sensors", "valve", "gauge"
        ):
            raise ValueError(f"Unknown LCD source: {source}")

        now = time.monotonic()

        if observed_at is None:
            if (
                source == "aruco"
                and data
                and data.get("captured_at") is not None
            ):
                observed_at = data["captured_at"]

            elif (
                source in ("valve", "gauge")
                and data
                and data.get("received_at_unix_s") is not None
            ):
                age = max(
                    0,
                    time.time() - data["received_at_unix_s"],
                )
                observed_at = now - age

            else:
                observed_at = now

        with self._lock:
            self._latest[source] = (
                deepcopy(data),
                observed_at,
            )

    @staticmethod
    def _font(size):
        from PIL import ImageFont

        try:
            from fonts.ttf import RobotoMedium

            return ImageFont.truetype(RobotoMedium, size)
        except (ImportError, OSError):
            try:
                return ImageFont.truetype("DejaVuSans.ttf", size)
            except OSError:
                return ImageFont.load_default()

    def start(self):
        if not self.enabled:
            return

        if self._display is not None:
            self._check_owner()
            return

        import st7735

        self._owner = threading.get_ident()

        self._display = st7735.ST7735(
            port=0,
            cs=1,
            dc="GPIO9",
            backlight="GPIO12",
            rotation=270,
            spi_speed_hz=10000000,
        )
        self._display.begin()

        self._history = deque(
            self._history,
            maxlen=self._display.width,
        )

    def _check_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError(
                "Only the LCD owner thread may draw "
                "or close the display"
            )

    def _canvas(self):
        from PIL import Image, ImageDraw

        image = Image.new(
            "RGB",
            (self._display.width, self._display.height),
            (6, 16, 25),
        )
        return image, ImageDraw.Draw(image)

    def _text(
        self,
        draw,
        text,
        y,
        size=17,
        colour=(255, 255, 255),
    ):
        text = str(text)
        font = self._font(size)
        width = self._display.width

        # Fit text inside the small display.
        while text:
            left, top, right, bottom = draw.textbbox(
                (0, 0), text, font=font
            )
            if right - left <= width - 8:
                break

            if size > 10:
                size -= 1
                font = self._font(size)
            else:
                text = text[:-1]

        left, top, right, _ = draw.textbbox(
            (0, 0), text, font=font
        )
        draw.text(
            (
                (width - (right - left)) / 2 - left,
                y - top,
            ),
            text,
            font=font,
            fill=colour,
        )

    def show(self, title, value):
        """Compatibility with existing simple display calls."""
        if self._display is None:
            return

        self._check_owner()
        image, draw = self._canvas()

        self._text(
            draw, title, 8, 11, (180, 205, 220)
        )
        self._text(draw, value, 37)
        self._display.display(image)

    def render(self):
        """Render the selected page using the latest readings."""
        if self._display is None:
            return

        self._check_owner()
        now = time.monotonic()

        with self._lock:
            latest = deepcopy(self._latest)

        def reading(source, max_age):
            data, observed = latest.get(
                source, (None, None)
            )

            if observed is None:
                return None, "WAITING"

            if now - observed > max_age:
                return None, "STALE"

            return data, "OK"

        sensors, sensor_status = reading("sensors", 5.0)
        sensors = sensors or {}

        # Proximity tap cycles through the three pages.
        proximity = sensors.get("proximity_counts")

        if proximity is not None:
            near = proximity > 1500

            if (
                near
                and not self._was_near
                and now - self._last_tap > 0.7
            ):
                self._page = (
                    self._page + 1
                ) % len(self.PAGES)
                self._last_tap = now

            self._was_near = near

        temperature = sensors.get("temperature_adjusted_c")

        if (
            temperature is not None
            and not math.isfinite(temperature)
        ):
            temperature = None

        sensor_timestamp = latest.get(
            "sensors", (None, None)
        )[1]

        # Add each sensor sample once.
        if (
            temperature is not None
            and sensor_timestamp != self._last_temperature_at
        ):
            self._history.append(temperature)
            self._last_temperature_at = sensor_timestamp

        page = self.PAGES[self._page]

        if page == "ip":
            ip, status = reading("ip", 15.0)
            self.show(
                "IP ADDRESS",
                (ip or "NO NETWORK") if status == "OK" else status,
            )
            return

        if page == "temperature":
            if sensor_status != "OK" or temperature is None:
                self.show(
                    "TEMPERATURE",
                    sensor_status
                    if sensor_status != "OK"
                    else "SENSOR ERROR",
                )
                return

            image, draw = self._canvas()
            draw.rectangle(
                (0, 0, image.width, image.height),
                fill="white",
            )
            draw.text(
                (2, 2),
                f"Temp: {temperature:.1f} C",
                font=self._font(11),
                fill="black",
            )

            values = list(self._history)
            low, high = min(values), max(values)
            span = max(high - low, 0.5)
            low = (low + high - span) / 2

            points = [
                (
                    index,
                    image.height - 2 - int(
                        (value - low)
                        / span
                        * (image.height - 27)
                    ),
                )
                for index, value in enumerate(values)
            ]

            if len(points) > 1:
                draw.line(points, fill="black", width=1)
            else:
                draw.point(points[0], fill="black")

            self._display.display(image)
            return

        # Combined page: only fresh detections produce rows.
        image, draw = self._canvas()
        rows = []

        def detected_result(source):
            result, status = reading(
                source, self.stale_seconds
            )

            if status != "OK" or not result:
                return None

            if result.get("status") in (
                "CAMERA ERROR", "CAMERA STOPPED"
            ):
                return None

            return result

        # ArUco markers.
        marker = detected_result("aruco")

        if marker:
            ids = marker.get("ids") or []

            if ids:
                # Cycle multiple markers every two seconds.
                index = int(now / 2) % len(ids)
                rows.append(f"ArUco: ID {ids[index]}")

        # Valve state and confidence.
        valve = detected_result("valve")

        if valve:
            detections = valve.get("detections") or []

            if detections:
                index = int(now / 2) % len(detections)
                detection = detections[index]

                state = detection.get("state") or {
                    "Valve-open": "open",
                    "Valve-closed": "closed",
                }.get(detection.get("label"), "unknown")

                text = f"Valve: {str(state).upper()}"
                confidence = detection.get("confidence")

                if (
                    isinstance(confidence, (int, float))
                    and math.isfinite(confidence)
                    and 0 <= confidence <= 1
                ):
                    text += f" {confidence:.0%}"

                rows.append(text)

        # Gauge pressure, when a valid reading is available.
        gauge = detected_result("gauge")

        if gauge:
            detections = gauge.get("detections") or []

            if detections:
                index = int(now / 2) % len(detections)
                detection = detections[index]
                pressure = detection.get("pressure") or {}

                # A gauge may be detected without a usable reading.
                text = "Gauge: DETECTED"
                value = pressure.get("value")
                unit = pressure.get("unit")

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None

                if (
                    value is not None
                    and math.isfinite(value)
                    and unit
                    and pressure.get("status", "valid") == "valid"
                ):
                    text = f"Gauge: {value:.1f} {unit}"

                rows.append(text)

        # Spread the detected categories over the available screen.
        if rows:
            row_height = image.height / len(rows)
            font_size = 17 if len(rows) == 1 else 13

            for index, text in enumerate(rows):
                y = int(
                    index * row_height
                    + (row_height - font_size) / 2
                )
                self._text(
                    draw,
                    text,
                    y=y,
                    size=font_size,
                    colour=(36, 220, 110),
                )

        # Refresh even when empty, clearing previous detections.
        self._display.display(image)

    def close(self):
        if self._display is not None:
            self._check_owner()

            try:
                self._display.set_backlight(0)
            finally:
                self._display = None
                self._owner = None