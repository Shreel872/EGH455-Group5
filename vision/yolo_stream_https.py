#!/usr/bin/env python3
"""Run the valve detector behind the matching Caddy HTTPS reverse proxy.

Copy the converted RVC2 NNArchive to models/valve_yolov5n_rvc2.tar.xz, then run:
    python yolo_stream_https.py

Open https://10.88.48.146:8443/ after trusting Caddy's local CA.
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import tarfile
import time

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork


CLASSES = ("Valve-closed", "Valve-open")
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "valve_yolov5n_rvc2.tar.xz"


def archive_classes(path: Path) -> tuple[str, ...]:
    """Read the model's class names before starting the OAK pipeline."""
    with tarfile.open(path, "r:xz") as archive:
        config_file = archive.extractfile("config.json")
        if config_file is None:
            raise ValueError("The model archive has no config.json")
        config = json.load(config_file)
    heads = config["model"]["heads"]
    if len(heads) != 1 or heads[0]["parser"] != "YOLOExtendedParser":
        raise ValueError("Expected one YOLOExtendedParser model head")
    return tuple(heads[0]["metadata"]["classes"])


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"RVC2 .tar.xz archive (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--http-port", type=int, default=8082)
    parser.add_argument("--websocket-port", type=int, default=8443)
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        parser.error(f"Model archive does not exist: {model_path}")
    if not model_path.name.endswith(".tar.xz"):
        parser.error("--model must be a converted RVC2 .tar.xz archive, not a .pt file")
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    try:
        classes = archive_classes(model_path)
    except (OSError, tarfile.TarError, KeyError, TypeError, ValueError) as error:
        parser.error(f"Cannot read valve model archive: {error}")
    if classes != CLASSES:
        parser.error(
            f"Wrong model classes: {list(classes)!r}. Expected {list(CLASSES)!r}."
        )

    remote = dai.RemoteConnection(
        address="127.0.0.1",
        webSocketPort=args.websocket_port,
        httpPort=args.http_port,
    )

    with dai.Pipeline() as pipeline:
        camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        network = pipeline.create(ParsingNeuralNetwork).build(
            camera, dai.NNArchive(str(model_path)), fps=args.fps
        )
        network.getParser(dai.node.DetectionParser).setConfidenceThreshold(args.conf)
        network.input.setBlocking(False)

        # Encode video on the OAK to avoid sending raw frames through the Pi.
        video = camera.requestOutput(
            (640, 640), dai.ImgFrame.Type.NV12, fps=args.fps
        )
        encoder = pipeline.create(dai.node.VideoEncoder)
        encoder.setDefaultProfilePreset(
            args.fps, dai.VideoEncoderProperties.Profile.H264_MAIN
        )
        encoder.setBitrateKbps(1500)
        video.link(encoder.input)
        remote.addTopic("images", encoder.out, "img")
        remote.addTopic("detections", network.out, "img")
        detections = network.out.createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()
        remote.registerPipeline(pipeline)
        print(f"Model: {model_path}")
        print(f"classes: {list(classes)}")
        print(f"Confidence threshold: {args.conf:.2f}")
        print("Open https://10.88.48.146:8443/ in Chrome")
        print("Press Ctrl+C to stop.")

        last_report = 0.0
        try:
            while pipeline.isRunning():
                packet = detections.tryGet()
                now = time.monotonic()
                if packet is not None and now - last_report >= 1.0:
                    if packet.detections:
                        results = [
                            f"{classes[d.label]} {d.confidence:.0%}"
                            for d in packet.detections
                        ]
                        print("Detected: " + ", ".join(results), flush=True)
                    else:
                        print("No valve detected", flush=True)
                    last_report = now

                if remote.waitKey(1) == ord("q"):
                    break
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
