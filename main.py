import cv2
import depthai as dai
import numpy as np


def getFrame(queue):
    """
    Extract frames from queue
    :param queue:
    :return:
    """
    frame = queue.get()
    return frame.getCvFrame()


def getMonoCamera(pipeline, isLeft):
    """
    Configure Mono Camera
    :param pipeline:
    :param isLeft:
    :return:
    """
    mono = pipeline.createMonoCamera()
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    if isLeft:
        # Get Left Camera
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        # Get Right Camera
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    return mono


def stereoPair(pipeline, monoLeft, monoRight):
    """
    Generates a stereo node
    :param pipeline:
    :param monoLeft:
    :param monoRight:
    :return:
    """
    # configure stereo pair for depth
    stereo = pipeline.createStereoDepth()

    # checks occluded pixels and marks them as invalid
    stereo.setLeftRightCheck(True)

    # Configure left and right cameras to work as stereo
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    return stereo


def mouseCallBack(event, x, y, flags, param):
    """
    record the pixel coordinate of a point when clicked
    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    :return:
    """
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y


if __name__ == "__main__":
    mouseX = 0
    mouseY = 640

    # Pipeline definition
    pipeline = dai.Pipeline()

    # Setting Up left and right mono cameras
    monoLeft = getMonoCamera(pipeline, isLeft=True)
    monoRight = getMonoCamera(pipeline, isLeft=False)

    # Stereo pair:
    stereo = stereoPair(pipeline, monoLeft, monoRight)

    # CREATE X-link out nodes and assign streams
    xoutdisp = pipeline.createXLinkOut()
    xoutdisp.setStreamName("disparity")

    xoutRectifiedLeft = pipeline.createXLinkOut()
    xoutRectifiedLeft.setStreamName("rectifiedLeft")

    xoutRectifiedRight = pipeline.createXLinkOut()
    xoutRectifiedRight.setStreamName("rectifiedRight")

    # Link stereodepths
    stereo.disparity.link(xoutdisp.input)
    stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
    stereo.rectifiedRight.link(xoutRectifiedRight.input)

    # MOVING PIPELINE TO OAK DEE
    with dai.Device(pipeline) as device:
        # Obtain Queues
        disparityQueue = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)
        RectifiedLeftQueue = device.getOutputQueue(name="rectifiedLeft", maxSize=1, blocking=False)
        RectifiedRightQueue = device.getOutputQueue(name="rectifiedRight", maxSize=1, blocking=False)

        disparityMultiplier = 255 / stereo.initialConfig.getMaxDisparity()
        cv2.namedWindow("Stereo Pair")
        cv2.setMouseCallback("Stereo Pair", mouseCallBack)
        sideBySide = False

        while True:
            # Get disparity map
            disparity = getFrame(disparityQueue)
            disparity = (disparity * disparityMultiplier).astype(np.uint8)
            disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

            # Get left and right rectified frames
            leftFrame = getFrame(RectifiedLeftQueue)
            rightFrame = getFrame(RectifiedRightQueue)

            if sideBySide:
                # Show side by side view:
                imOut = np.hstack((leftFrame, rightFrame))
            else:
                imOut = np.uint8(leftFrame/2 + rightFrame/2)

            # Convert to RGB
            imOut = cv2.cvtColor(imOut, cv2.COLOR_GRAY2RGB)

            # Draw scan line
            imOut = cv2.circle(imOut, (mouseX, mouseY), 2, (0,0,255), 2)

            # Draw clicked point
            imOut = cv2.circle(imOut, (mouseX, mouseY), 2, (255, 255, 128), 2)

            cv2.imshow("Stereo Pair", imOut)
            cv2.imshow("Disparity", disparity)

            # Check for keyboard input
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("t"):
                sideBySide = not sideBySide



