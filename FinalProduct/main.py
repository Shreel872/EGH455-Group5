"""Coordinate camera, detection, sensors, LCD, JSONL and MJPEG output."""

from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
from itertools import count
import logging
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time

from ArucoDetector import ArucoWorker
from cameraService import CameraService
from EventStorage import Event, EventStorage
from EventUploader import EventUploader
from LCDcontroller import LCDController
from ObjectDetection import ObjectDetection
from visionOutput import VisionOutput


LOG = logging.getLogger(__name__)

from collections import deque
from pathlib import Path
import math


class SensorReadings:
    """Read existing sensor objects; no camera or LCD dependencies.

    Call read() from one owning thread.
    """

    def __init__(self, bme280=None, proximity=None, factor=2.25):
        self.bme280 = bme280
        self.proximity = proximity
        self.factor = float(factor)

        if not math.isfinite(self.factor) or self.factor <= 0:
            raise ValueError("Temperature correction factor must be positive")

        self._cpu_history = deque(maxlen=5)

    @staticmethod
    def _cpu_temperature():
        path = Path("/sys/class/thermal/thermal_zone0/temp")
        return float(path.read_text(encoding="ascii").strip()) / 1000.0

    def read(self):
        errors = {}

        def read_value(name, getter):
            if getter is None:
                errors[name] = "Sensor or reading unavailable"
                return None

            try:
                value = float(getter())
                if not math.isfinite(value):
                    raise ValueError("Non-finite sensor reading")
                return value
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                errors[name] = str(exc)
                return None

        cpu = read_value(
            "cpu_temperature_c",
            self._cpu_temperature,
        )
        raw = read_value(
            "temperature_raw_c",
            getattr(self.bme280, "get_temperature", None),
        )
        humidity = read_value(
            "humidity_percent",
            getattr(self.bme280, "get_humidity", None),
        )
        pressure = read_value(
            "ambient_pressure_hpa",
            getattr(self.bme280, "get_pressure", None),
        )
        proximity = read_value(
            "proximity_counts",
            getattr(self.proximity, "get_proximity", None),
        )

        adjusted = None

        if cpu is not None:
            # Match the original script's initial five-sample history.
            if not self._cpu_history:
                self._cpu_history.extend([cpu] * 5)
            else:
                self._cpu_history.append(cpu)

        if cpu is not None and raw is not None:
            average_cpu = sum(self._cpu_history) / len(self._cpu_history)
            adjusted = raw - ((average_cpu - raw) / self.factor)
        else:
            errors["temperature_adjusted_c"] = (
                "Requires current CPU and sensor temperatures"
            )

        return {
            "cpu_temperature_c": cpu,
            "temperature_raw_c": raw,
            "temperature_adjusted_c": adjusted,
            "temperature_compensation_factor": self.factor,
            "humidity_percent": humidity,
            "ambient_pressure_hpa": pressure,
            "proximity_counts": proximity,
            "errors": errors,
        }
def load_sensors():
    proximity = None
    temperature = None

    try:
        try:
            from ltr559 import LTR559
            proximity = LTR559()
        except ImportError:
            import ltr559
            proximity = ltr559

    except (ImportError, OSError, RuntimeError):
        LOG.warning(
            "Proximity/light sensor unavailable",
            exc_info=True,
        )

    try:
        from bme280 import BME280
        temperature = BME280()

    except (ImportError, OSError, RuntimeError):
        LOG.warning(
            "BME280 unavailable",
            exc_info=True,
        )

    return proximity, temperature


def get_ip():
    try:
        with socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        ) as connection:
            connection.settimeout(0.2)
            connection.connect(("8.8.8.8", 80))

            return connection.getsockname()[0]

    except OSError:
        return "NO NETWORK"


def service_command(action):
    """Optionally release/restore the existing IP-display service."""
    try:
        result = subprocess.run(
            ["sudo", "-n", "systemctl", action, "ip-display"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode:
            LOG.warning(
                "ip-display %s failed: %s",
                action,
                result.stderr.strip(),
            )

        return result.returncode == 0

    except (OSError, subprocess.TimeoutExpired) as exc:
        LOG.warning(
            "Cannot %s ip-display: %s",
            action,
            exc,
        )
        return False


class FinalProduct:
    def __init__(self, args):
        self.args = args
        self.stop = threading.Event()

        self.storage = EventStorage(args.events)

        self.uploader = EventUploader(
            url=args.upload_url,
            ack_path=args.events.with_suffix(
                args.events.suffix + ".uploaded.json"
            ),
        )

        self.lcd = LCDController(
            enabled=args.live and not args.no_lcd,
            stale_seconds=args.stale_seconds,
        )

        # Created only in live mode.
        self.camera = None
        self.objects = None
        self.aruco = None
        self.output = None
        self.sensor_reader = None

        self.frame_ids = count()
        self.threads = []

        self.lock = threading.Lock()
        self.latest = {}
        self.failure = None

        self.closed = False
        self.restart_ip_display = False

    def publish(self, source, data, observed_at=None):
        """Update the latest observation without writing to disk.

        observed_at uses time.monotonic().
        """
        if observed_at is None:
            observed_at = time.monotonic()

        age = max(
            0,
            time.monotonic() - observed_at,
        )

        observed_utc = datetime.fromtimestamp(
            time.time() - age,
            timezone.utc,
        ).isoformat()

        with self.lock:
            self.latest[source] = (
                deepcopy(data),
                observed_at,
                observed_utc,
            )

    def snapshot(self):
        """Combine the latest readings, retaining their individual ages."""
        now = time.monotonic()

        with self.lock:
            latest = deepcopy(self.latest)

        limits = {
            "sensors": 5.0,
            "device": 15.0,
            "aruco": self.args.stale_seconds,
            "valve": self.args.stale_seconds,
            "gauge": self.args.stale_seconds,
        }

        result = {
            "mode": "live" if self.args.live else "dry_run",
        }

        for source, limit in limits.items():
            if source not in latest:
                result[source] = {
                    "freshness": "unavailable",
                    "observed_at": None,
                    "age_s": None,
                    "data": None,
                }
                continue

            data, observed, observed_utc = latest[source]
            age = max(0, now - observed)

            result[source] = {
                "freshness": (
                    "stale" if age > limit else "fresh"
                ),
                "observed_at": observed_utc,
                "age_s": round(age, 3),
                "data": data,
            }

        return result

    def record(self):
        self.storage.append(
            Event(
                event_type="snapshot",
                value=self.snapshot(),
                source="FinalProduct",
                metadata={"schema_version": 1},
            )
        )

    def handle_frame(self, frame):
        """Submit clean full-view frames to the ArUco worker."""
        self.aruco.submit(
            frame,
            next(self.frame_ids),
            time.monotonic(),
        )

    def handle_detection(self, source, frame, metadata):
        """Receive valve/gauge results already produced by the VPU."""
        result = self.objects.process(source, metadata)

        for detection in result["detections"]:
            if source == "valve":
                detection["state"] = {
                    "Valve-open": "open",
                    "Valve-closed": "closed",
                }.get(
                    detection["label"],
                    "unknown",
                )

            else:
                # Preserve an actual pressure reading when supplied.
                # Do not substitute confidence or ambient pressure.
                if detection.get("pressure") is None:
                    detection["pressure"] = {
                        "value": None,
                        "unit": None,
                        "status": "not_available",
                    }

        result["status"] = (
            "detected"
            if result["detections"]
            else "not_detected"
        )

        age = max(
            0,
            time.time() - result["received_at_unix_s"],
        )
        observed = time.monotonic() - age

        self.publish(source, result, observed)

        # update() is thread-safe; it does not draw on the LCD.
        self.lcd.update(
            source,
            result,
            observed_at=observed,
        )

        self.output.submit(source, frame, result)

    def worker(self, function):
        """Make background failures visible to the coordinator."""
        try:
            function()

        except Exception as exc:
            LOG.exception(
                "Worker failed: %s",
                threading.current_thread().name,
            )

            with self.lock:
                if self.failure is None:
                    self.failure = exc

            self.stop.set()

    def start_thread(self, name, function):
        thread = threading.Thread(
            target=self.worker,
            args=(function,),
            name=name,
        )

        thread.start()
        self.threads.append(thread)

    def camera_loop(self):
        self.camera.run(
            on_frame=self.handle_frame,
            stop_requested=self.stop.is_set,
            on_detection=self.handle_detection,
        )

        if not self.stop.is_set():
            raise RuntimeError(
                "Camera pipeline stopped unexpectedly"
            )

    def record_loop(self):
        while not self.stop.is_set():
            self.record()
            self.stop.wait(self.args.record_interval)

    def upload_loop(self):
        while not self.stop.is_set():
            sent = self.uploader.upload(
                self.storage.pending(),
                stop=self.stop,
            )

            if sent:
                LOG.info(
                    "Uploaded %d new snapshots",
                    sent,
                )

            self.stop.wait(self.args.upload_interval)

    def run(self):
        if not self.args.live:
            self.record()

            LOG.info(
                "Dry-run snapshot saved to %s; "
                "no hardware or upload started",
                self.args.events,
            )
            return

        self.objects = ObjectDetection(
            valve_model=self.args.model,
            gauge_model=self.args.gauge_model,
            valve_confidence=self.args.confidence,
            gauge_confidence=self.args.gauge_confidence,
            valve_resize=self.args.valve_resize,
            gauge_resize=self.args.gauge_resize,
        )

        self.camera = CameraService(
            width=self.args.width,
            height=self.args.height,
            fps=self.args.fps,
            models=self.objects.model_specs(),
        )

        self.aruco = ArucoWorker(
            dictionary=self.args.dictionary,
        )

        self.output = VisionOutput(
            sources=self.objects.sources(),
            directory=self.args.vision_record,
            upload_url=self.args.video_upload_url,
            max_fps=self.args.output_fps,
        )

        self.objects.start()
        self.aruco.start()
        self.output.start()

        if (
            self.args.manage_ip_display
            and not self.args.no_lcd
        ):
            try:
                active = subprocess.run(
                    [
                        "systemctl",
                        "is-active",
                        "--quiet",
                        "ip-display",
                    ],
                    timeout=5,
                ).returncode == 0

            except (OSError, subprocess.TimeoutExpired):
                active = False

            if active:
                self.restart_ip_display = service_command(
                    "stop"
                )

        # LCD start/render/close all run on this main thread.
        self.lcd.start()

        proximity, temperature = load_sensors()

        self.sensor_reader = SensorReadings(
            bme280=temperature,
            proximity=proximity,
        )

        self.start_thread(
            "camera-acquisition",
            self.camera_loop,
        )

        self.start_thread(
            "event-recorder",
            self.record_loop,
        )

        if self.args.upload_url:
            self.start_thread(
                "event-uploader",
                self.upload_loop,
            )

        deadline = (
            time.monotonic() + self.args.duration
            if self.args.duration
            else float("inf")
        )

        last_marker = None
        last_ip_at = -float("inf")

        while (
            not self.stop.is_set()
            and time.monotonic() < deadline
        ):
            started = time.monotonic()

            self.output.check()

            # Read sensors regardless of the selected LCD page.
            readings = self.sensor_reader.read()
            observed = time.monotonic()

            self.publish(
                "sensors",
                readings,
                observed,
            )

            self.lcd.update(
                "sensors",
                readings,
                observed_at=observed,
            )

            marker = self.aruco.latest_result()

            if (
                marker is not None
                and marker["frame_id"] != last_marker
            ):
                last_marker = marker["frame_id"]

                marker["status"] = (
                    "DETECTED"
                    if marker["ids"]
                    else "NO MARKER"
                    if marker["locked"]
                    else "SEARCHING"
                )

                self.publish(
                    "aruco",
                    marker,
                    marker["captured_at"],
                )

                self.lcd.update("aruco", marker)

            if started - last_ip_at >= 5:
                ip = get_ip()

                self.publish(
                    "device",
                    {"ip_address": ip},
                )

                self.lcd.update("ip", ip)
                last_ip_at = started

            self.lcd.render()

            elapsed = time.monotonic() - started

            self.stop.wait(
                max(
                    0,
                    1 / self.args.refresh_hz - elapsed,
                )
            )

        with self.lock:
            failure = self.failure

        if failure is not None:
            raise RuntimeError(
                "A product worker failed"
            ) from failure

    def close(self):
        """Clean up once, including after a partial startup."""
        if self.closed:
            return

        self.closed = True
        self.stop.set()

        errors = []

        for thread in self.threads:
            thread.join(timeout=15)

            if thread.is_alive():
                errors.append(
                    f"{thread.name} has not stopped"
                )

        components = (
            (self.aruco, "stop"),
            (self.output, "close"),
            (self.objects, "close"),
            (self.lcd, "close"),
        )

        for component, method in components:
            if component is None:
                continue

            try:
                getattr(component, method)()
            except Exception as exc:
                errors.append(str(exc))

        if self.restart_ip_display:
            service_command("start")

        if errors:
            raise RuntimeError(
                "Shutdown problems: " + "; ".join(errors)
            )


def parse_args():
    parser = ArgumentParser(description=__doc__)
    models_dir = (
    Path(__file__).resolve().parent.parent / "vision" / "models"
    )
    parser.add_argument("--live", action="store_true")

    parser.add_argument(
        "--model",
        type=Path,
        default=models_dir / "valve_yolov5n_rvc2.tar.xz",
        help="Converted valve NNArchive",
    )
    parser.add_argument(
        "--gauge-model", 
        type=Path,
        default= models_dir/"yolov8n_gauge_old.rvc2.tar.xz",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--gauge-confidence",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--valve-resize",
        choices=("crop", "stretch", "letterbox"),
        default="crop",
    )
    parser.add_argument(
        "--gauge-resize",
        choices=("crop", "stretch", "letterbox"),
        default="stretch",
    )

    parser.add_argument("--dictionary", default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=15)

    parser.add_argument(
        "--refresh-hz",
        type=float,
        default=15,
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=30,
    )
    parser.add_argument(
        "--stale-seconds",
        type=float,
        default=2,
    )

    parser.add_argument(
        "--record-interval",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--upload-interval",
        type=float,
        default=5,
    )

    parser.add_argument(
        "--events",
        type=Path,
        default=Path("data/events.jsonl"),
    )
    parser.add_argument(
        "--upload-url",
        help="JSON event receiver",
    )
    parser.add_argument(
        "--video-upload-url",
        help="Optional compatible MJPEG receiver",
    )
    parser.add_argument(
        "--vision-record",
        type=Path,
        help="Optional JPEG/metadata test directory",
        default= "data/vision-test",
    )

    parser.add_argument("--no-lcd", action="store_true")
    parser.add_argument(
        "--manage-ip-display",
        action="store_true",
        help="Stop an active ip-display service and restore on exit",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Seconds to run; zero runs until stopped",
    )

    args = parser.parse_args()

    if args.live and not args.model:
        parser.error("--live requires --model")

    positive_values = (
        args.width,
        args.height,
        args.fps,
        args.refresh_hz,
        args.output_fps,
        args.stale_seconds,
        args.record_interval,
        args.upload_interval,
    )

    if min(positive_values) <= 0 or args.duration < 0:
        parser.error(
            "Rates and intervals must be positive; "
            "duration cannot be negative"
        )

    return args


def main():
    subprocess.run(["sudo", "systemctl", "stop", "ip-display"], check=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    product = FinalProduct(parse_args())

    def stop_requested(_signal, _frame):
        product.stop.set()

    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)

    try:
        product.run()
    finally:
        product.close()
        subprocess.run(["sudo", "systemctl", "start", "ip-display"], check=False)


if __name__ == "__main__":
    main()