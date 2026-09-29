"""One OAK camera pipeline for valve and gauge detection."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import tarfile
import time


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    confidence: float = 0.4
    resize: str = "crop"


def model_config(spec: ModelSpec):
    """Read the input geometry and labels from a converted NNArchive."""
    if spec.resize not in ("crop", "stretch", "letterbox"):
        raise ValueError("Unknown resize mode")

    if not 0 <= spec.confidence <= 1:
        raise ValueError("Confidence must be between zero and one")

    with tarfile.open(spec.path) as archive:
        try:
            member = archive.extractfile("config.json")
        except KeyError as exc:
            raise ValueError(
                f"{spec.path} is not a converted NNArchive: "
                "config.json is missing"
            ) from exc

        if member is None:
            raise ValueError("Missing model configuration")

        config = json.load(member)["model"]

    if len(config["inputs"]) != 1 or len(config["heads"]) != 1:
        raise ValueError("Expected one image input and one detection head")

    image = config["inputs"][0]
    head = config["heads"][0]

    if image["layout"] != "NCHW" or image["shape"][:2] != [1, 3]:
        raise ValueError("Expected a 1x3xHxW input")

    if head["parser"] != "YOLOExtendedParser":
        raise ValueError("Expected a YOLOExtendedParser archive")

    height, width = image["shape"][2:]

    if min(height, width) <= 0:
        raise ValueError("Invalid model dimensions")

    return (
        width,
        height,
        image["preprocessing"]["dai_type"],
        head["metadata"]["classes"],
    )


class SequencePairer:
    """Match detections with their exact frame, with bounded buffering."""

    def __init__(self, limit=8):
        self.frames = OrderedDict()
        self.detections = OrderedDict()
        self.limit = limit
        self.last = -1

    def add(self, kind, packet):
        sequence = int(packet.getSequenceNum())

        if sequence <= self.last:
            return

        cache = self.frames if kind == "frame" else self.detections
        cache[sequence] = packet

        while len(cache) > self.limit:
            cache.popitem(last=False)

    def newest(self):
        shared = self.frames.keys() & self.detections.keys()

        if not shared:
            return None

        sequence = max(shared)
        pair = self.frames[sequence], self.detections[sequence]

        for cache in (self.frames, self.detections):
            for key in list(cache):
                if key <= sequence:
                    del cache[key]

        self.last = sequence
        return pair


class CameraService:
    def __init__(
        self,
        width=1280,
        height=720,
        fps=15.0,
        models=(),
    ):
        if min(width, height, fps) <= 0:
            raise ValueError("Camera dimensions and FPS must be positive")

        self.width = width
        self.height = height
        self.fps = fps
        self.models = tuple(models)

        if len({model.name for model in self.models}) != len(self.models):
            raise ValueError("Each model needs a unique name")

    def run(self, on_frame, stop_requested, on_detection):
        """
        on_frame(frame):
            Optional clean BGR frames for your ArUco worker.

        on_detection(source, frame, metadata):
            Matching model-input frame and detection results.

        Callbacks should submit work and return promptly.
        """
        import depthai as dai
        from depthai_nodes.node import ParsingNeuralNetwork

        configs = [
            (spec, model_config(spec))
            for spec in self.models
        ]

        with dai.Pipeline() as pipeline:
            camera = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_A
            )

            # Optional full-view frames for the ArUco worker.
            raw_queue = None

            if on_frame is not None:
                raw = camera.requestOutput(
                    (self.width, self.height),
                    dai.ImgFrame.Type.BGR888p,
                    resizeMode=dai.ImgResizeMode.LETTERBOX,
                    fps=self.fps,
                )

                raw_queue = raw.createOutputQueue(
                    maxSize=2,
                    blocking=False,
                )

            streams = []

            for spec, config in configs:
                width, height, pixel_type, labels = config

                inference = camera.requestOutput(
                    (width, height),
                    getattr(dai.ImgFrame.Type, pixel_type),
                    resizeMode=getattr(
                        dai.ImgResizeMode,
                        spec.resize.upper(),
                    ),
                    fps=self.fps,
                )

                network = pipeline.create(ParsingNeuralNetwork).build(
                    inference,
                    dai.NNArchive(str(spec.path)),
                )

                network.getParser(
                    dai.node.DetectionParser
                ).setConfidenceThreshold(spec.confidence)

                network.input.setBlocking(False)

                frame_queue = network.passthrough.createOutputQueue(
                    maxSize=4,
                    blocking=False,
                )

                detection_queue = network.out.createOutputQueue(
                    maxSize=4,
                    blocking=False,
                )

                streams.append(
                    (
                        spec,
                        labels,
                        frame_queue,
                        detection_queue,
                        SequencePairer(),
                    )
                )

            pipeline.start()

            while pipeline.isRunning() and not stop_requested():
                if raw_queue is not None:
                    packet = raw_queue.tryGet()

                    if packet is not None:
                        on_frame(packet.getCvFrame())

                for spec, labels, frames, detections, pairer in streams:
                    # Bounded draining avoids starving other tasks.
                    for kind, queue in (
                        ("frame", frames),
                        ("detection", detections),
                    ):
                        for _ in range(4):
                            packet = queue.tryGet()

                            if packet is None:
                                break

                            pairer.add(kind, packet)

                    pair = pairer.newest()

                    if pair is None:
                        continue

                    image, prediction = pair
                    frame = image.getCvFrame()

                    results = []

                    for detection in prediction.detections:
                        class_id = int(detection.label)

                        label = (
                            labels[class_id]
                            if 0 <= class_id < len(labels)
                            else str(class_id)
                        )

                        results.append(
                            {
                                "class_id": class_id,
                                "label": label,
                                "confidence": float(
                                    detection.confidence
                                ),
                                "bbox_xyxy": [
                                    float(detection.xmin),
                                    float(detection.ymin),
                                    float(detection.xmax),
                                    float(detection.ymax),
                                ],
                            }
                        )

                    metadata = {
                        "schema_version": 1,
                        "source": spec.name,
                        "frame_id": int(image.getSequenceNum()),
                        "device_timestamp_s": (
                            image.getTimestampDevice().total_seconds()
                        ),
                        "received_at_unix_s": time.time(),
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                        "resize": spec.resize,
                        "coordinate_space": "normalized_model_input",
                        "detections": results,
                    }

                    on_detection(spec.name, frame, metadata)

                time.sleep(0.005)