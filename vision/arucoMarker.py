#!/usr/bin/env python3
"""Detect and display ArUco markers from an OAK RGB camera."""

from argparse import ArgumentParser
from http import server
import threading
import time

import cv2
import depthai as dai
import numpy as np


class StreamState:
    def __init__(self):
        self.frame = None
        self.sequence = 0
        self.condition = threading.Condition()

    def update(self, jpeg):
        with self.condition:
            self.frame = jpeg
            self.sequence += 1
            self.condition.notify_all()


class StreamHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            body = (
                "<!doctype html><html><head><title>ArUco detection</title>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<style>body{margin:0;background:#111;color:#eee;font:16px Arial,sans-serif}"
                "main{max-width:1280px;margin:auto;padding:16px}"
                "img{display:block;width:100%;height:auto;background:#222}"
                "p{color:#aaa}</style></head><body><main>"
                "<h1>ArUco detection</h1><img src='/stream' alt='Camera stream'>"
                "<p>Green boxes mark decoded ArUco IDs. The status is drawn on the video.</p>"
                "</main></body></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path != "/stream":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last_sequence = 0
        try:
            while True:
                with self.server.stream_state.condition:
                    fresh = self.server.stream_state.condition.wait_for(
                        lambda: self.server.stream_state.sequence > last_sequence,
                        timeout=5,
                    )
                    if not fresh:
                        continue
                    last_sequence = self.server.stream_state.sequence
                    jpeg = self.server.stream_state.frame
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, format, *args):
        return


class StreamServer(server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, stream_state):
        self.stream_state = stream_state
        super().__init__(address, StreamHandler)


def create_detector(dictionary_name):
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
        raise RuntimeError("ArUco is unavailable. Install a recent opencv-contrib-python build.")
    name = dictionary_name.upper().removeprefix("DICT_")
    key = f"DICT_{name}"
    if not hasattr(cv2.aruco, key):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, key))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


class DetectorSearch:
    """Search common dictionaries, then keep using the first stable match."""

    def __init__(self, dictionary_name):
        self.auto = dictionary_name.lower() == "auto"
        if self.auto:
            names = [f"{bits}X{bits}_{size}" for bits in (4, 5, 6, 7)
                     for size in (50, 100, 250, 1000)]
            names.append("ARUCO_ORIGINAL")
        else:
            names = [dictionary_name.upper().removeprefix("DICT_")]
        self.detectors = [(name, create_detector(name)) for name in names]
        self.index = 0
        self.locked_index = None if self.auto else 0
        self.probe = None
        self.probe_count = 0
        self.misses = 0

    def detect(self, gray):
        index = self.locked_index if self.locked_index is not None else self.index
        name, detector = self.detectors[index]
        corners, ids, rejected = detector.detectMarkers(gray)
        if ids is not None:
            marker_ids = tuple(int(value) for value in ids.flatten())
            if self.locked_index is None:
                probe = (index, marker_ids)
                self.probe_count = self.probe_count + 1 if probe == self.probe else 1
                self.probe = probe
                if self.probe_count >= 2:
                    self.locked_index = index
                    print(f"Locked ArUco dictionary: {name}")
            self.misses = 0
        elif self.locked_index is not None and self.auto:
            self.misses += 1
            if self.misses >= 30:
                self.locked_index = None
                self.index = (index + 1) % len(self.detectors)
                self.probe = None
                self.probe_count = 0
        else:
            self.index = (index + 1) % len(self.detectors)
            self.probe = None
            self.probe_count = 0
        return corners, ids, rejected, name, self.locked_index is not None


def draw_detections(frame, detector_search, show_rejected=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected, dictionary_name, locked = detector_search.detect(gray)
    marker_ids = [] if ids is None else [int(marker_id) for marker_id in ids.flatten()]

    status = f"DETECTED: {len(marker_ids)} marker(s)" if marker_ids else "NO MARKER DETECTED"
    mode = "Dictionary" if locked else "Searching"
    details = f"{mode}: {dictionary_name}   Rejected candidates: {len(rejected)}"
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1] - 1, 650), 76), (25, 25, 25), -1)
    cv2.putText(frame, status, (14, 31), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 220, 0) if marker_ids else (0, 110, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, details, (14, 61), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (230, 230, 230), 1, cv2.LINE_AA)

    if show_rejected:
        for candidate in rejected:
            points = np.rint(candidate).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [points], True, (0, 165, 255), 1)

    for marker_id, marker_corners in zip(marker_ids, corners):
        points = np.rint(marker_corners).astype(np.int32).reshape(4, 2)
        cv2.polylines(frame, [points.reshape(-1, 1, 2)], True, (0, 255, 0), 3)
        x, y, width, height = cv2.boundingRect(points)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        label = f"ArUco ID {marker_id}"
        label_y = y - 8 if y >= 30 else y + height + 24
        cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return marker_ids, len(rejected), dictionary_name


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", default="auto",
                        help="ArUco dictionary, e.g. 4X4_50; default auto searches standard families")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--http-port", type=int, default=8082)
    parser.add_argument("--show-rejected", action="store_true",
                        help="Draw candidate quads that could not be decoded in orange")
    args = parser.parse_args()
    detector_search = DetectorSearch(args.dictionary)

    stream_state = StreamState()
    stream_server = StreamServer(("0.0.0.0", args.http_port), stream_state)
    server_thread = threading.Thread(target=stream_server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Open http://<computer-ip>:{args.http_port}/ in a browser")
    print(f"ArUco dictionary: {args.dictionary}; output {args.width}x{args.height}")

    try:
        with dai.Pipeline() as pipeline:
            camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            output = camera.requestOutput(
                (args.width, args.height),
                dai.ImgFrame.Type.BGR888p,
                dai.ImgResizeMode.LETTERBOX,
                args.fps,
            )
            frame_queue = output.createOutputQueue(maxSize=2, blocking=False)
            pipeline.start()
            last_ids = None
            last_report = 0.0
            while pipeline.isRunning():
                frame = frame_queue.get().getCvFrame()
                cpu_temp = get_cpu_temperature()
                marker_ids, rejected_count, dictionary_name = draw_detections(
                    frame, detector_search, args.show_rejected
                )
                now = time.monotonic()
                if marker_ids != last_ids or (not marker_ids and now - last_report >= 5):
                    print(f"ArUco IDs: {marker_ids or 'none'}; rejected candidates: {rejected_count}")
                    last_ids = marker_ids
                    last_report = now
                success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    stream_state.update(encoded.tobytes())
    except KeyboardInterrupt:
        print("Stopping ArUco stream.")
    finally:
        stream_server.shutdown()
        stream_server.server_close()


if __name__ == "__main__":
    main()
