"""ArUco adapter used by the product coordinator."""

from __future__ import annotations

from typing import Any


class ArucoDetector:
    """Search dictionaries, confirm a match, then track that dictionary."""

    def __init__(self, dictionary: str = "auto") -> None:
        self.dictionary = dictionary
        self._cv2: Any = None
        self._detectors = []
        self._auto = dictionary.lower() == "auto"
        self._next_index = 0
        self._candidate = None
        self._locked = None
        self._misses = 0

    def start(self) -> None:
        import cv2

        if not hasattr(cv2, "aruco") or not hasattr(
            cv2.aruco, "ArucoDetector"
        ):
            raise RuntimeError(
                "This Python environment needs OpenCV with ArUco support"
            )

        if self._auto:
            names = [
                f"{bits}X{bits}_{size}"
                for bits in (4, 5, 6, 7)
                for size in (50, 100, 250, 1000)
            ]
            names.append("ARUCO_ORIGINAL")
        else:
            names = [self.dictionary.upper().removeprefix("DICT_")]

        detectors = []

        for name in names:
            key = f"DICT_{name}"
            if not hasattr(cv2.aruco, key):
                raise ValueError(f"Unknown ArUco dictionary: {name}")

            dictionary = cv2.aruco.getPredefinedDictionary(
                getattr(cv2.aruco, key)
            )
            parameters = cv2.aruco.DetectorParameters()
            parameters.cornerRefinementMethod = (
                cv2.aruco.CORNER_REFINE_SUBPIX
            )

            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            detectors.append((name, detector))

        self._cv2 = cv2
        self._detectors = detectors
        self._next_index = 0
        self._candidate = None
        self._locked = None if self._auto else 0
        self._misses = 0

    def process(self, frame: Any) -> dict[str, Any]:
        if not self._detectors:
            raise RuntimeError("Call ArucoDetector.start() first")

        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)

        if self._locked is not None:
            index = self._locked
        elif self._candidate is not None:
            index = self._candidate[0]
        else:
            index = self._next_index

        name, detector = self._detectors[index]
        _, ids, rejected = detector.detectMarkers(gray)

        found = (
            ()
            if ids is None
            else tuple(sorted(int(value) for value in ids.flatten()))
        )
        confirmed = ()

        if self._locked is not None:
            self._misses = 0 if found else self._misses + 1
            confirmed = found

            # Resume dictionary searching after 30 processed misses.
            if self._auto and self._misses >= 30:
                self._locked = None
                self._candidate = None
                self._next_index = (index + 1) % len(self._detectors)
                self._misses = 0

        elif found and self._candidate == (index, found):
            # Same dictionary and IDs on two consecutive processed frames.
            self._locked = index
            self._candidate = None
            self._misses = 0
            confirmed = found

        elif found:
            self._candidate = (index, found)

        else:
            self._candidate = None
            self._next_index = (index + 1) % len(self._detectors)

        return {
            "ids": list(confirmed),
            "dictionary": name,
            "rejected": len(rejected) if rejected is not None else 0,
            "locked": self._locked is not None,
        }

    def close(self) -> None:
        self._detectors = []
        self._cv2 = None
        self._candidate = None
        self._locked = None
        self._misses = 0


# Compatibility with the earlier spelling.
ArucoDector = ArucoDetector
from copy import deepcopy
import threading
import time
from typing import Any


class ArucoWorker:
    """Run ArucoDetector in one background thread.

    Each instance can be started once.
    Create a new instance if you need to restart it.
    """

    def __init__(self, dictionary: str = "auto") -> None:
        self._detector = ArucoDetector(dictionary)

        self._condition = threading.Condition()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

        self._stopping = False
        self._pending: tuple[Any, int, float] | None = None
        self._result: dict[str, Any] | None = None
        self._error: Exception | None = None

    def _check_error(self) -> None:
        """Call while holding self._condition."""
        if self._error is not None:
            raise RuntimeError("ArUco worker failed") from self._error

    def start(self) -> None:
        """Start the worker and wait for detector initialization."""
        with self._condition:
            if self._thread is not None or self._stopping:
                raise RuntimeError("This ArucoWorker cannot be started again")

            self._thread = threading.Thread(
                target=self._run,
                name="aruco-detection",
                daemon=False,
            )
            self._thread.start()

        self._ready.wait()

        with self._condition:
            self._check_error()

    def submit(
        self,
        frame: Any,
        frame_id: int,
        captured_at: float | None = None,
    ) -> None:
        """Submit a BGR image without waiting for detection.

        Replaces any frame that is waiting to be processed.
        captured_at should use time.monotonic().
        """
        if captured_at is None:
            captured_at = time.monotonic()

        # Own a separate image so other tasks can modify their frame.
        owned_frame = frame.copy()

        with self._condition:
            self._check_error()

            if self._thread is None or not self._ready.is_set():
                raise RuntimeError("Call start() before submit()")
            if self._stopping:
                raise RuntimeError("ArUco worker is stopping")

            self._pending = (owned_frame, frame_id, captured_at)
            self._condition.notify()

    def latest_result(self) -> dict[str, Any] | None:
        """Return a snapshot, or None before the first completed detection."""
        with self._condition:
            self._check_error()

            # Prevent readers from modifying the worker's saved result.
            return deepcopy(self._result)

    def stop(self, timeout: float = 5.0) -> None:
        """Discard waiting work and wait for current detection to finish."""
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
            thread = self._thread

        if thread is not None:
            thread.join(timeout)

            if thread.is_alive():
                raise TimeoutError(
                    "ArUco worker is still finishing its current detection"
                )

    def _run(self) -> None:
        try:
            # The worker owns the detector's entire lifecycle.
            self._detector.start()
            self._ready.set()

            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stopping or self._pending is not None
                    )

                    if self._stopping:
                        return

                    frame, frame_id, captured_at = self._pending
                    self._pending = None

                # Expensive work happens outside the shared lock.
                started_at = time.monotonic()
                detection = self._detector.process(frame)
                completed_at = time.monotonic()

                result = {
                    **detection,
                    "frame_id": frame_id,
                    "captured_at": captured_at,
                    "completed_at": completed_at,
                    "processing_ms": (completed_at - started_at) * 1000,
                    "latency_ms": (completed_at - captured_at) * 1000,
                }

                with self._condition:
                    if not self._stopping:
                        # Also publish empty IDs when no marker is detected.
                        self._result = result

        except Exception as exc:
            with self._condition:
                self._error = exc
                self._stopping = True
                self._pending = None
                self._condition.notify_all()

        finally:
            self._detector.close()

            # Unblock start() even if initialization failed.
            self._ready.set()
    