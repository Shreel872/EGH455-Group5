import depthai as dai
 
with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    nn = pipeline.create(dai.node.DetectionNetwork).build(
        cam, dai.NNModelDescription("yolov6-nano"), fps=15.0)
    nn.setConfidenceThreshold(0.5)
    nn.input.setBlocking(False)
 
    labels = nn.getClasses() or []
    q = nn.out.createOutputQueue()
 
    pipeline.start()
    while pipeline.isRunning():
        dets = q.get()                      # ImgDetections
        for d in dets.detections:
            name = labels[d.label] if d.label < len(labels) else str(d.label)
            # xmin/ymin/xmax/ymax are normalised 0..1
            print(f"{name:<15} {d.confidence:.2f} "
                  f"({d.xmin:.2f},{d.ymin:.2f})-({d.xmax:.2f},{d.ymax:.2f})")
 