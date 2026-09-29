"""Manage valve and gauge detections from CameraService."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

from cameraService import ModelSpec


class ObjectDetection:
    def __init__(
        self,
        valve_model: str | Path,
        gauge_model: str | Path | None = None,
        valve_confidence: float = 0.4,
        gauge_confidence: float = 0.4,
        valve_resize: str = "crop",
        gauge_resize: str = "stretch",
    ) -> None:
        if not valve_model:
            raise ValueError("A converted valve model is required")

        self._models = {
            "valve": ModelSpec(
                name="valve",
                path=Path(valve_model).expanduser(),
                confidence=valve_confidence,
                resize=valve_resize,
            )
        }

        if gauge_model:
            self._models["gauge"] = ModelSpec(
                name="gauge",
                path=Path(gauge_model).expanduser(),
                confidence=gauge_confidence,
                resize=gauge_resize,
            )

        for model in self._models.values():
            if (
                not math.isfinite(model.confidence)
                or not 0 <= model.confidence <= 1
            ):
                raise ValueError(
                    f"Invalid confidence for {model.name}"
                )

            if model.resize not in ("crop", "stretch", "letterbox"):
                raise ValueError(
                    f"Invalid resize mode for {model.name}"
                )

        self._started = False

    def model_specs(self) -> tuple[ModelSpec, ...]:
        """Return both model configurations for CameraService."""
        return tuple(self._models.values())

    def sources(self) -> tuple[str, ...]:
        """Names of the enabled detection feeds."""
        return tuple(self._models)

    def start(self) -> None:
        """Validate paths; CameraService builds the actual VPU nodes."""
        for model in self._models.values():
            if not model.path.is_file():
                raise FileNotFoundError(
                    f"{model.name} model does not exist: {model.path}"
                )

        self._started = True

    def process(
        self,
        source: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Accept either valve or gauge results from CameraService.

        Returns metadata even when no objects are detected.
        Does not open the camera or repeat inference.
        """
        if not self._started:
            raise RuntimeError(
                "ObjectDetection.start() must be called first"
            )

        if source not in self._models:
            raise ValueError(f"Unknown detection source: {source}")

        if metadata.get("source") != source:
            raise ValueError("Detection source does not match metadata")

        threshold = self._models[source].confidence

        result = deepcopy(metadata)
        accepted = []

        for detection in result["detections"]:
            confidence = float(detection["confidence"])

            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Invalid detection confidence")

            if confidence >= threshold:
                detection["confidence"] = confidence
                accepted.append(detection)

        result["detections"] = accepted

        return result

    def close(self) -> None:
        """CameraService remains responsible for device shutdown."""
        self._started = False