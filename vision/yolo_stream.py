#!/usr/bin/env python3
from argparse import ArgumentParser
import depthai as dai

p = ArgumentParser()
p.add_argument("--model", default="yolov6-nano")
p.add_argument("--fps", type=float, default=30.0)
p.add_argument("--conf", type=float, default=0.5)
p.add_argument("--webSocketPort", type=int, default=8765)
p.add_argument("--httpPort", type=int, default=8082)
args = p.parse_args()

remote = dai.RemoteConnection(address="0.0.0.0",
                              webSocketPort=args.webSocketPort,
                              httpPort=args.httpPort)

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    model = (dai.NNArchive(args.model) if args.model.endswith(".tar.xz")
             else dai.NNModelDescription(args.model))

    nn = pipeline.create(dai.node.DetectionNetwork).build(cam, model, fps=args.fps)
    nn.setConfidenceThreshold(args.conf)
    nn.input.setBlocking(False)

    print("classes:", nn.getClasses())

    vis_out = cam.requestOutput((512, 288), dai.ImgFrame.Type.NV12, fps=args.fps)

    enc = pipeline.create(dai.node.VideoEncoder)
    enc.setDefaultProfilePreset(args.fps, dai.VideoEncoderProperties.Profile.H264_MAIN)
    enc.setBitrateKbps(2000)
    vis_out.link(enc.input)
    remote.addTopic("detections", nn.out, "img")
    remote.addTopic("images", enc.out, "img")

    pipeline.start()
    remote.registerPipeline(pipeline)
    print("USB:", pipeline.getDefaultDevice().getUsbSpeed())
    print(f"visualizer on port {args.httpPort}")

    while pipeline.isRunning():
        if remote.waitKey(1) == ord("q"):
            break