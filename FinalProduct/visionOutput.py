"""Background JPEG/MJPEG output. Does not start a server."""

from copy import deepcopy
import json
import logging
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen


LOG = logging.getLogger(__name__)

CONTENT_TYPE = "multipart/x-mixed-replace; boundary=frame"


def mjpeg_part(jpeg, metadata):
    """Build one JPEG part of a multipart MJPEG stream."""
    metadata_header = json.dumps(
        metadata,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")

    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: "
        + str(len(jpeg)).encode("ascii")
        + b"\r\nX-Vision-Metadata: "
        + metadata_header
        + b"\r\n\r\n"
        + jpeg
        + b"\r\n"
    )


def encode_annotated(frame, metadata, quality):
    import cv2

    image = frame.copy()
    height, width = image.shape[:2]

    for detection in metadata["detections"]:
        x1, y1, x2, y2 = detection["bbox_xyxy"]

        x1, x2 = [
            max(0, min(width - 1, round(x * width)))
            for x in (x1, x2)
        ]

        y1, y2 = [
            max(0, min(height - 1, round(y * height)))
            for y in (y1, y2)
        ]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"{detection['label']} "
            f"{detection['confidence']:.0%}"
        )

        cv2.putText(
            image,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    success, buffer = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if not success:
        raise RuntimeError("JPEG encoding failed")

    return buffer.tobytes()


class VisionOutput:
    def __init__(
        self,
        sources=("valve", "gauge"),
        directory=None,
        upload_url=None,
        max_fps=15,
        quality=90,
        encoder=encode_annotated,
    ):
        if max_fps <= 0 or not 1 <= quality <= 100:
            raise ValueError("Invalid output FPS or JPEG quality")

        self.sources = frozenset(sources)

        if not self.sources or any(
            not source.isidentifier()
            for source in self.sources
        ):
            raise ValueError("Sources must be safe identifier names")

        self.directory = Path(directory) if directory else None
        self.upload_url = upload_url
        self.period = 1 / max_fps
        self.quality = quality
        self.encoder = encoder

        self.condition = threading.Condition()
        self.pending = {}
        self.latest = {}
        self.versions = {}

        self.error = None
        self.last_network_error = None
        self.stopping = False
        self.thread = None

    def start(self):
        with self.condition:
            if self.thread is not None or self.stopping:
                raise RuntimeError(
                    "Create a new VisionOutput to restart"
                )

            if self.directory:
                self.directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            self.thread = threading.Thread(
                target=self._run,
                name="vision-output",
                daemon=False,
            )

            self.thread.start()

    def check(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError(
                    "Vision output worker failed"
                ) from self.error

    def submit(self, source, frame, metadata):
        if (
            source not in self.sources
            or metadata["source"] != source
        ):
            raise ValueError("Unregistered or mismatched source")

        # Own a copy before other tasks modify the original frame.
        item = frame.copy(), deepcopy(metadata)

        with self.condition:
            self.check()

            if self.thread is None or self.stopping:
                raise RuntimeError("Output worker is not running")

            # Replace older waiting work for this source.
            self.pending[source] = item
            self.condition.notify()

    def snapshot(self, source):
        """Return a matching JPEG/metadata pair, or None."""
        with self.condition:
            self.check()
            item = self.latest.get(source)

            if item is None:
                return None

            return item[0], deepcopy(item[1])

    def iter_mjpeg(self, source):
        """
        Yield a continuous multipart MJPEG body.

        An external consumer can use this iterator.
        This method does not start any server.
        """
        if source not in self.sources:
            raise ValueError("Unknown source")

        version = 0

        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: (
                        self.stopping
                        or self.error is not None
                        or self.versions.get(source, 0) > version
                    )
                )

                self.check()

                if self.stopping:
                    break

                version = self.versions[source]
                jpeg, metadata = self.latest[source]
                metadata = deepcopy(metadata)

            yield mjpeg_part(jpeg, metadata)

        yield b"--frame--\r\n"

    def close(self):
        with self.condition:
            self.stopping = True
            self.pending.clear()
            self.condition.notify_all()
            thread = self.thread

        if thread is not None:
            thread.join(timeout=15)

            if thread.is_alive():
                raise TimeoutError(
                    "Output worker has not finished shutting down"
                )

        self.check()

    def _record(self, source, jpeg, metadata, payload):
        files = (
            ("jpg", jpeg),
            ("json", json.dumps(metadata, indent=4).encode()),
            ("multipart", payload),
        )

        for suffix, content in files:
            target = self.directory / f"{source}_latest.{suffix}"
            temporary = target.with_suffix(target.suffix + ".tmp")

            temporary.write_bytes(content)
            temporary.replace(target)

        with (self.directory / "detections.jsonl").open("a") as log:
            log.write(json.dumps(metadata) + "\n")

    def _upload(self, source, payload):
        url = self.upload_url.format(source=source)

        request = Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": CONTENT_TYPE,
                "X-Vision-Source": source,
            },
        )

        with urlopen(request, timeout=3) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(
                    f"Upload returned HTTP {response.status}"
                )

    def _run(self):
        next_output = 0.0
        retry_after = {}

        try:
            while True:
                with self.condition:
                    self.condition.wait_for(
                        lambda: self.stopping or bool(self.pending)
                    )

                    if self.stopping:
                        return

                    remaining = next_output - time.monotonic()

                    if remaining > 0:
                        self.condition.wait(timeout=remaining)
                        continue

                    batch = self.pending
                    self.pending = {}

                for source, (frame, metadata) in batch.items():
                    with self.condition:
                        if self.stopping:
                            return

                    jpeg = self.encoder(
                        frame,
                        metadata,
                        self.quality,
                    )

                    metadata["encoded_at_unix_s"] = time.time()

                    # Finite payload for recording or an outgoing POST.
                    payload = (
                        mjpeg_part(jpeg, metadata)
                        + b"--frame--\r\n"
                    )

                    with self.condition:
                        self.latest[source] = jpeg, metadata
                        self.versions[source] = (
                            self.versions.get(source, 0) + 1
                        )
                        self.condition.notify_all()

                    if self.directory:
                        self._record(
                            source,
                            jpeg,
                            metadata,
                            payload,
                        )

                    if (
                        self.upload_url
                        and time.monotonic()
                        >= retry_after.get(source, 0)
                    ):
                        try:
                            self._upload(source, payload)

                            with self.condition:
                                self.last_network_error = None

                        except Exception as exc:
                            retry_after[source] = (
                                time.monotonic() + 2
                            )

                            with self.condition:
                                self.last_network_error = str(exc)

                            LOG.warning(
                                "Upload failed for %s: %s",
                                source,
                                exc,
                            )

                next_output = time.monotonic() + self.period

        except Exception as exc:
            with self.condition:
                self.error = exc
                self.stopping = True
                self.condition.notify_all()