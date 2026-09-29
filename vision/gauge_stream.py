import depthai as dai

MODEL = "yolo_Gauge.tar.xz"  # model name after installing/registering the model

with dai.Pipeline() as pipeline:

    # OAK-D Lite RGB camera
    cam = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_A
    )

    # YOLO detection network
    nn = pipeline.create(dai.node.DetectionNetwork).build(
        cam,
        dai.NNModelDescription(MODEL),
        fps=15.0
    )

    nn.setConfidenceThreshold(0.5)
    nn.input.setBlocking(False)

    # Get class names from the model
    labels = nn.getClasses() or []

    # Detection output
    q = nn.out.createOutputQueue(maxSize=4, blocking=False)

    pipeline.start()

    while pipeline.isRunning():
        dets = q.get()

        for d in dets.detections:
            name = (
                labels[d.label]
                if d.label < len(labels)
                else str(d.label)
            )

            print(
                f"{name:<15} "
                f"{d.confidence:.2f} "
                f"({d.xmin:.2f},{d.ymin:.2f})-"
                f"({d.xmax:.2f},{d.ymax:.2f})"
            )
