import depthai as dai

MODEL_PATH = "models/mobile.blob/mobile_openvino_2021.4_6shave.blob"

pipeline = dai.Pipeline()
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(MODEL_PATH)
cam = pipeline.create(dai.node.ColorCamera)
cam.out.link(nn.input)

# Send NN out to the host via XLink
nnXout = pipeline.create(dai.node.XLinkOut)
nnXout.setStreamName("nn")
nn.out.link(nnXout.input)

with dai.Device(pipeline) as Device:
    qNn = Device.getOutputQueue("nn")

    nnData = qNn.get()  # Blocking

    # NN can output from multiple layers. Print all layer names:
    print(nnData.getAllLayerNames())
