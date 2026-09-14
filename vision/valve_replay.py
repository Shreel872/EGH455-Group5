#!/usr/bin/env python3
"""
Run the converted valve model on a recorded video through the real OAK
pipeline -- no live scene needed. Tests the superblob, the NN archive and
on-device parsing exactly as they'll run on the day.

    python valve_replay.py --model valve.tar.xz --video test.mp4
    # then browse to http://<pi-ip>:8082
"""
from argparse import ArgumentParser
from pathlib import Path
import depthai as dai

p = ArgumentParser()
p.add_argument("--model", required=True, help="path to NNArchive .tar.xz")
p.add_argument("--video", required=True, help="path to test video")
p.add_argument("--conf", type=float, default=0.4)
p.add_argument("--webSocketPort", type=int, default=8765)
p.add_argument("--httpPort", type=int, default=8082)
args = p.parse_args()

remote = dai.RemoteConnection(address="0.0.0.0",
                              webSocketPort=args.webSocketPort,
                              httpPort=args.httpPort)

with dai.Pipeline() as pipeline:
    replay = pipeline.create(dai.node.ReplayVideo)
    replay.setReplayVideoFile(Path(args.video))

    nn = pipeline.create(dai.node.DetectionNetwork).build(
        replay, dai.NNArchive(args.model))
    nn.setConfidenceThreshold(args.conf)

    print("classes:", nn.getClasses())

    remote.addTopic("detections", nn.out, "img")
    remote.addTopic("images", replay.out, "img")

    pipeline.start()
    remote.registerPipeline(pipeline)
    print("USB:", pipeline.getDefaultDevice().getUsbSpeed())

    while pipeline.isRunning():
        if remote.waitKey(1) == ord("q"):
            break