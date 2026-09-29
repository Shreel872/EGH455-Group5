#!/usr/bin/env python3
"""Run the valve detector on an OAK-D Lite and show its detections in a browser.

Copy the converted RVC2 NNArchive to models/valve_yolov5n_rvc2.tar.xz, then run:
    python yolo_stream.py

Open http://<pi-ip>:8082/ on a computer on the same network.
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import tarfile
import time
import zoneinfo
from datetime import datetime, timezone, timedelta, tzinfo, date
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork


CLASSES = ("Valve-closed", "Valve-open")
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "valve_yolov5n_rvc2.tar.xz"


def archive_config(path: Path) -> dict:
    """Read the model input and parser configuration before starting the pipeline."""
    with tarfile.open(path, "r:xz") as archive:
        config_file = archive.extractfile("config.json")
        if config_file is None:
            raise ValueError("The model archive has no config.json")
        config = json.load(config_file)
    heads = config["model"]["heads"]
    if len(heads) != 1 or heads[0]["parser"] != "YOLOExtendedParser":
        raise ValueError("Expected one YOLOExtendedParser model head")
    return config


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
    parser.add_argument(
        "--resize", choices=("stretch", "crop", "letterbox"), default="stretch",
        help="Use stretch for the current square-stretched training dataset; crop reproduces the old view",
    )
    parser.add_argument("--http-port", type=int, default=8082)
    parser.add_argument("--websocket-port", type=int, default=8765)
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
        config = archive_config(model_path)
        classes = tuple(config["model"]["heads"][0]["metadata"]["classes"])
        inputs = config["model"]["inputs"]
        if len(inputs) != 1 or inputs[0]["layout"] != "NCHW":
            raise ValueError("Expected one NCHW image input")
        batch, channels, height, width = inputs[0]["shape"]
        if batch != 1 or channels != 3 or width <= 0 or height <= 0:
            raise ValueError("Expected a positive-size 1x3xHxW image input")
        frame_type = getattr(dai.ImgFrame.Type, inputs[0]["preprocessing"]["dai_type"])
    except (OSError, tarfile.TarError, KeyError, TypeError, ValueError, AttributeError) as error:
        parser.error(f"Cannot read valve model archive: {error}")
    if classes != CLASSES:
        parser.error(
            f"Wrong model classes: {list(classes)!r}. Expected {list(CLASSES)!r}."
        )

    remote = dai.RemoteConnection(
        address="0.0.0.0",
        webSocketPort=args.websocket_port,
        httpPort=args.http_port,
    )

    with dai.Pipeline() as pipeline:
        camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        resize_mode = getattr(dai.ImgResizeMode, args.resize.upper())
        # Match the training geometry explicitly; preserve the archive's colour order.
        inference = camera.requestOutput(
            (width, height), frame_type, resizeMode=resize_mode, fps=args.fps
        )
        network = pipeline.create(ParsingNeuralNetwork).build(
            inference, dai.NNArchive(str(model_path))
        )
        network.getParser(dai.node.DetectionParser).setConfidenceThreshold(args.conf)
        network.input.setBlocking(False)

        # Keep the displayed field of view and aspect ratio identical to inference.
        video = camera.requestOutput(
            (width, height), dai.ImgFrame.Type.NV12, resizeMode=resize_mode, fps=args.fps
        )
        encoder = pipeline.create(dai.node.VideoEncoder)
        encoder.setDefaultProfilePreset(
            args.fps, dai.VideoEncoderProperties.Profile.MJPEG
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
        print(f"Input: {width}x{height}, {frame_type}, resize={args.resize}")
        print(f"Confidence threshold: {args.conf:.2f}")
        print(f"Framerate of the video stream: {args.fps:.1f} FPS")
        print(f"Video quality: {encoder.getQuality()}")
        print(f"Open http://<pi-ip>:{args.http_port}/ in a browser")
        print("Press Ctrl+C to stop.")

        last_report = time.monotonic()
        detection_frames = 0
        class_frames = [0] * len(classes)
        peak_confidence = [0.0] * len(classes)
        try:
            while pipeline.isRunning():
                packet = detections.tryGet()
                now = time.monotonic()
                if packet is not None:
                    detection_frames += 1
                    for label in {d.label for d in packet.detections}:
                        class_frames[label] += 1
                        print(f"class_frames: {class_frames}")
                    for detection in packet.detections:
                        peak_confidence[detection.label] = max(
                            peak_confidence[detection.label], detection.confidence
                        )
                if class_frames[1] > 5 or class_frames[0] > 5: #This to ensure the detection is not a false positive
                    print(f"Allocated detection Period reached, preparing to save data");
                    timestamp = time.time()
                    aest_time = datetime.fromtimestamp(timestamp, tz=zoneinfo.ZoneInfo("Australia/Brisbane"))
                    numberOfFrames = ( class_frames[1] if class_frames[1] > 5 else class_frames[0] );
                    if class_frames[1] == 0:
                        detection_type = "Valve-open"
                    else:
                        detection_type = "Valve-closed"
                    data_to_save = {
                        "time": aest_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                        "Amount of Frames": numberOfFrames,
                        "Detected": detection_type,
                    }
                    with open("data.json", "w") as json_file:
                        json.dump(data_to_save, json_file, indent=4)
                        print(f"data saved to json: {data_to_save}")
                if now - last_report >= 2.0:
                    inference_fps = detection_frames / (now - last_report)
                    results = [
                        f"{name}: {class_frames[i]}/{detection_frames} frames, "
                        f"peak {peak_confidence[i]:.0%}"
                        for i, name in enumerate(classes)
                    ]
                    print(f"{inference_fps:.1f} inference FPS | " + " | ".join(results), flush=True)
                    class_frames = [0] * len(classes)
                    peak_confidence = [0.0] * len(classes)
                    last_report = now
                    detection_frames = 0

                if remote.waitKey(1) == ord("q"):
                    break
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
