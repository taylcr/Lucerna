#!/usr/bin/env python3
"""
All-in-One DepthAI Example
– Displays a color camera feed with spatial-location warnings (if an object is too near),
– Shows a blended undistorted RGB/depth image (with adjustable blend and FPS display),
– Runs feature tracking (with motion estimation) on the left mono camera,
– Runs MobileNet‑SSD object detection on the right mono camera.
"""

import cv2
import depthai as dai
import numpy as np
import math
import time
from datetime import timedelta
from collections import deque
from pathlib import Path
import sys

# ---------------------------- User Parameters ----------------------------

# For spatial warnings (in millimeters)
WARNING = 500   # e.g. 500mm = 50cm (orange rectangle)
CRITICAL = 300  # e.g. 300mm = 30cm (red rectangle)

# FPS for some pipelines
FPS = 30.0

# For MobileNet SSD: default blob path (you may pass a different path as first argument)
nnPath = str((Path(__file__).parent / Path(r'models\mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())
if len(sys.argv) > 1:
    nnPath = sys.argv[1]
if not Path(nnPath).exists():
    raise FileNotFoundError(f"Required blob file not found: {nnPath}")

# MobileNetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Blend weights for RGB/depth window (global so that trackbar callback can update them)
rgbWeight = 0.4
depthWeight = 0.6

# ---------------------------- Utility Classes and Functions ----------------------------

class FPSCounter:
    def __init__(self):
        self.frameTimes = []

    def tick(self):
        now = time.time()
        self.frameTimes.append(now)
        self.frameTimes = self.frameTimes[-10:]

    def getFps(self):
        if len(self.frameTimes) <= 1:
            return 0
        return (len(self.frameTimes) - 1) / (self.frameTimes[-1] - self.frameTimes[0])


def updateBlendWeights(percentRgb):
    """Callback for trackbar to update blending weights."""
    global rgbWeight, depthWeight
    rgbWeight = float(percentRgb) / 100.0
    depthWeight = 1.0 - rgbWeight


def colorizeDepth(frameDepth):
    """Colorizes a depth frame using a logarithmic scale and a JET colormap."""
    invalidMask = frameDepth == 0
    try:
        valid = frameDepth[frameDepth != 0]
        if len(valid) == 0:
            minDepth = 0
            maxDepth = 0
        else:
            minDepth = np.percentile(valid, 3)
            maxDepth = np.percentile(valid, 95)
        logDepth = np.log(frameDepth, where=frameDepth != 0)
        logMinDepth = np.log(minDepth) if minDepth > 0 else 0
        logMaxDepth = np.log(maxDepth) if maxDepth > 0 else 1
        np.nan_to_num(logDepth, copy=False, nan=logMinDepth)
        logDepth = np.clip(logDepth, logMinDepth, logMaxDepth)
        depthFrameColor = np.interp(logDepth, (logMinDepth, logMaxDepth), (0, 255))
        depthFrameColor = np.nan_to_num(depthFrameColor)
        depthFrameColor = depthFrameColor.astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
        depthFrameColor[invalidMask] = 0
    except Exception as e:
        depthFrameColor = np.zeros((frameDepth.shape[0], frameDepth.shape[1], 3), dtype=np.uint8)
    return depthFrameColor


def frameNorm(frame, bbox):
    """
    Normalizes a bounding box (in 0..1 range) to image coordinates.
    bbox: tuple (xmin, ymin, xmax, ymax)
    """
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


# ---------------------------- Feature Tracking Classes ----------------------------

class CameraMotionEstimator:
    def __init__(self, filter_weight=0.5, motion_threshold=0.01, rotation_threshold=0.05):
        self.last_avg_flow = np.array([0.0, 0.0])
        self.filter_weight = filter_weight
        self.motion_threshold = motion_threshold
        self.rotation_threshold = rotation_threshold

    def estimate_motion(self, feature_paths):
        most_prominent_motion = "Camera Staying Still"
        max_magnitude = 0.0
        avg_flow = np.array([0.0, 0.0])
        total_rotation = 0.0
        vanishing_point = np.array([0.0, 0.0])
        num_features = len(feature_paths)

        # Debug print:
        print(f"Number of features: {num_features}")

        if num_features == 0:
            return most_prominent_motion, vanishing_point

        for path in feature_paths.values():
            if len(path) >= 2:
                src = np.array([path[-2].x, path[-2].y])
                dst = np.array([path[-1].x, path[-1].y])
                avg_flow += dst - src
                motion_vector = dst + (dst - src)
                vanishing_point += motion_vector
                rotation = np.arctan2(dst[1] - src[1], dst[0] - src[0])
                total_rotation += rotation

        avg_flow /= num_features
        avg_rotation = total_rotation / num_features
        vanishing_point /= num_features

        print(f"Average Flow: {avg_flow}")
        print(f"Average Rotation: {avg_rotation}")

        avg_flow = (self.filter_weight * self.last_avg_flow + (1 - self.filter_weight) * avg_flow)
        self.last_avg_flow = avg_flow

        flow_magnitude = np.linalg.norm(avg_flow)
        rotation_magnitude = abs(avg_rotation)

        if flow_magnitude > max_magnitude and flow_magnitude > self.motion_threshold:
            if abs(avg_flow[0]) > abs(avg_flow[1]):
                most_prominent_motion = 'Right' if avg_flow[0] < 0 else 'Left'
            else:
                most_prominent_motion = 'Down' if avg_flow[1] < 0 else 'Up'
            max_magnitude = flow_magnitude

        if rotation_magnitude > max_magnitude and rotation_magnitude > self.rotation_threshold:
            most_prominent_motion = 'Rotating'

        return most_prominent_motion, vanishing_point


class FeatureTrackerDrawer:
    lineColor = (200, 0, 200)
    pointColor = (0, 0, 255)
    vanishingPointColor = (255, 0, 255)  # Violet
    circleRadius = 2
    maxTrackedFeaturesPathLength = 30
    trackedFeaturesPathLength = 10

    trackedIDs = None
    trackedFeaturesPath = None

    direction_colors = {
        "Up": (0, 255, 255),    # Yellow
        "Down": (0, 255, 0),    # Green
        "Left": (255, 0, 0),    # Blue
        "Right": (0, 0, 255),   # Red
    }

    def __init__(self, windowName):
        self.windowName = windowName
        cv2.namedWindow(windowName)
        self.trackedIDs = set()
        self.trackedFeaturesPath = dict()

    def trackFeaturePath(self, features):
        newTrackedIDs = set()
        for currentFeature in features:
            currentID = currentFeature.id
            newTrackedIDs.add(currentID)

            if currentID not in self.trackedFeaturesPath:
                self.trackedFeaturesPath[currentID] = deque()

            path = self.trackedFeaturesPath[currentID]
            path.append(currentFeature.position)
            while len(path) > max(1, FeatureTrackerDrawer.trackedFeaturesPathLength):
                path.popleft()
            self.trackedFeaturesPath[currentID] = path

        featuresToRemove = set()
        for oldId in self.trackedIDs:
            if oldId not in newTrackedIDs:
                featuresToRemove.add(oldId)
        for id in featuresToRemove:
            self.trackedFeaturesPath.pop(id, None)
        self.trackedIDs = newTrackedIDs

    def drawVanishingPoint(self, img, vanishing_point):
        cv2.circle(img, (int(vanishing_point[0]), int(vanishing_point[1])), self.circleRadius,
                   self.vanishingPointColor, -1, cv2.LINE_AA, 0)

    def drawFeatures(self, img, vanishing_point=None, prominent_motion=None):
        if prominent_motion in self.direction_colors:
            point_color = self.direction_colors[prominent_motion]
        else:
            point_color = self.pointColor

        for featurePath in self.trackedFeaturesPath.values():
            path = featurePath
            for j in range(len(path) - 1):
                src = (int(path[j].x), int(path[j].y))
                dst = (int(path[j + 1].x), int(path[j + 1].y))
                cv2.line(img, src, dst, point_color, 1, cv2.LINE_AA, 0)
            if len(path) > 0:
                j = len(path) - 1
                cv2.circle(img, (int(path[j].x), int(path[j].y)), self.circleRadius,
                           point_color, -1, cv2.LINE_AA, 0)

        if prominent_motion:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(prominent_motion, font, font_scale, font_thickness)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            text_y = text_size[1] + 20  # 20 pixels from top
            text_color = self.direction_colors.get(prominent_motion, (255, 255, 255))
            cv2.putText(img, prominent_motion, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        if vanishing_point is not None:
            self.drawVanishingPoint(img, vanishing_point)


# ---------------------------- Build the Pipeline ----------------------------

pipeline = dai.Pipeline()

# --------- COLOR CAMERA (CAM_A) ---------
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setFps(FPS)
# For the high-resolution (ISP) output used in undistortion and blending:
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setIspScale(1, 3)  # Downscale ISP output if desired
# For spatial location overlay, use a low-res preview (e.g. 300x300)
camRgb.setPreviewSize(300, 300)

# XLinkOut for the color camera (used by spatial location overlay)
xoutColor = pipeline.create(dai.node.XLinkOut)
xoutColor.setStreamName("color")
camRgb.video.link(xoutColor.input)

# --------- MONO CAMERAS (LEFT and RIGHT) ---------
leftMono = pipeline.create(dai.node.MonoCamera)
leftMono.setBoardSocket(dai.CameraBoardSocket.LEFT)
leftMono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
leftMono.setFps(FPS)

rightMono = pipeline.create(dai.node.MonoCamera)
rightMono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
rightMono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
rightMono.setFps(FPS)

# --------- STEREO DEPTH ---------
stereo = pipeline.create(dai.node.StereoDepth)
stereo.initialConfig.setConfidenceThreshold(50)  # Updated API call
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(True)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)  # Use DEFAULT preset
# Align depth to the color camera (CAM_A)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

leftMono.out.link(stereo.left)
rightMono.out.link(stereo.right)

# --------- SPATIAL LOCATION CALCULATOR (for distance warnings) ---------
slc = pipeline.create(dai.node.SpatialLocationCalculator)
# Create a grid of ROIs (15 x 9) over the image
for x in range(15):
    for y in range(9):
        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 200
        config.depthThresholds.upperThreshold = 10000
        config.roi = dai.Rect(dai.Point2f((x + 0.5) * 0.0625, (y + 0.5) * 0.1),
                              dai.Point2f((x + 1.5) * 0.0625, (y + 1.5) * 0.1))
        config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
        slc.initialConfig.addROI(config)
stereo.depth.link(slc.inputDepth)

xoutSlc = pipeline.create(dai.node.XLinkOut)
xoutSlc.setStreamName("slc")
slc.out.link(xoutSlc.input)

# --------- SYNC NODE (to bundle RGB and aligned depth for blending) ---------
sync = pipeline.create(dai.node.Sync)
sync.setSyncThreshold(timedelta(seconds=0.5 / FPS))
camRgb.isp.link(sync.inputs["rgb"])
# Directly link the stereo depth (which is already aligned) to the sync node:
stereo.depth.link(sync.inputs["depth_aligned"])

xoutSync = pipeline.create(dai.node.XLinkOut)
xoutSync.setStreamName("out")
sync.out.link(xoutSync.input)

# --------- FEATURE TRACKER (on left mono camera) ---------
featureTracker = pipeline.create(dai.node.FeatureTracker)
featureTracker.setHardwareResources(2, 2)
leftMono.out.link(featureTracker.inputImage)

xoutTrackedFeatures = pipeline.create(dai.node.XLinkOut)
xoutTrackedFeatures.setStreamName("trackedFeaturesLeft")
featureTracker.outputFeatures.link(xoutTrackedFeatures.input)

xoutPassthroughLeft = pipeline.create(dai.node.XLinkOut)
xoutPassthroughLeft.setStreamName("passthroughLeft")
featureTracker.passthroughInputImage.link(xoutPassthroughLeft.input)

# --------- MOBILENET-SSD DETECTION (on right mono camera) ---------
manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(300, 300)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
rightMono.out.link(manip.inputImage)

nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(nnPath)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Link ImageManip output to both the neural network and an output for display.
manip.out.link(nn.input)

xoutRight = pipeline.create(dai.node.XLinkOut)
xoutRight.setStreamName("right")
manip.out.link(xoutRight.input)

xoutNN = pipeline.create(dai.node.XLinkOut)
xoutNN.setStreamName("nn")
nn.out.link(xoutNN.input)

# ---------------------------- Device Setup and Main Loop ----------------------------

with dai.Device(pipeline) as device:
    # Get calibration data for undistortion (for the color camera)
    calibrationHandler = device.readCalibration()
    # Use CAM_A for calibration since that's the board socket used for the color camera
    rgbDistortion = calibrationHandler.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)
    rgbIntrinsics = calibrationHandler.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 1280, 720)

    # Create output queues for all streams
    qColor = device.getOutputQueue(name="color", maxSize=4, blocking=False)
    qSlc = device.getOutputQueue(name="slc", maxSize=4, blocking=False)
    qSync = device.getOutputQueue(name="out", maxSize=8, blocking=False)
    qTrackedFeatures = device.getOutputQueue(name="trackedFeaturesLeft", maxSize=4, blocking=False)
    qPassthroughLeft = device.getOutputQueue(name="passthroughLeft", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    qNN = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    # Set up windows for blended RGB/depth and feature tracking.
    windowName = "rgb-depth"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 1280, 720)
    cv2.createTrackbar("RGB Weight %", windowName, int(rgbWeight * 100), 100, updateBlendWeights)

    left_window_name = "Left"
    featureDrawer = FeatureTrackerDrawer(left_window_name)
    cameraEstimator = CameraMotionEstimator(filter_weight=0.5, motion_threshold=0.3, rotation_threshold=0.5)

    fpsCounter = FPSCounter()

    while True:
        # 1. Spatial Location – get color camera (300x300 preview) and overlay ROI warnings
        inColor = qColor.tryGet()
        inSlc = qSlc.tryGet()

        if inColor is not None:
            colorFrame = inColor.getCvFrame()
        else:
            colorFrame = None

        if inSlc is not None and colorFrame is not None:
            slc_data = inSlc.getSpatialLocations()
            for depthData in slc_data:
                roi = depthData.config.roi
                roi = roi.denormalize(width=colorFrame.shape[1], height=colorFrame.shape[0])
                xmin = int(roi.topLeft().x)
                ymin = int(roi.topLeft().y)
                xmax = int(roi.bottomRight().x)
                ymax = int(roi.bottomRight().y)
                coords = depthData.spatialCoordinates
                distance = math.sqrt(coords.x ** 2 + coords.y ** 2 + coords.z ** 2)
                if distance == 0:
                    continue
                if distance < CRITICAL:
                    rect_color = (0, 0, 255)  # Red
                elif distance < WARNING:
                    rect_color = (0, 140, 255)  # Orange
                else:
                    continue  # Only draw for near objects
                cv2.rectangle(colorFrame, (xmin, ymin), (xmax, ymax), rect_color, 2)
                cv2.putText(colorFrame, "{:.1f}m".format(distance / 1000), (xmin + 10, ymin + 20),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, rect_color)
            cv2.imshow("Spatial Location", colorFrame)

        # 2. RGB/Depth Blending (using the sync node’s MessageGroup)
        msgGroup = qSync.tryGet()
        if msgGroup is not None:
            fpsCounter.tick()
            frameRgb = msgGroup.getMessage("rgb")
            frameDepth = msgGroup.getMessage("depth_aligned")
            if frameRgb is not None and frameDepth is not None:
                cvFrame = frameRgb.getCvFrame()
                # Undistort the RGB frame
                cvFrameUndistorted = cv2.undistort(cvFrame, np.array(rgbIntrinsics), np.array(rgbDistortion))
                alignedDepth = frameDepth.getFrame()
                alignedDepthColorized = colorizeDepth(alignedDepth)
                cv2.imshow("Depth aligned", alignedDepthColorized)
                blended = cv2.addWeighted(cvFrameUndistorted, rgbWeight, alignedDepthColorized, depthWeight, 0)
                cv2.putText(blended, f"FPS: {fpsCounter.getFps():.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(windowName, blended)

        # 3. Feature Tracking (using left mono passthrough and tracked features)
        inPassthroughLeft = qPassthroughLeft.tryGet()
        if inPassthroughLeft is not None:
            passthroughFrame = inPassthroughLeft.getFrame()  # grayscale image
            leftFrame = cv2.cvtColor(passthroughFrame, cv2.COLOR_GRAY2BGR)
            inTrackedFeatures = qTrackedFeatures.tryGet()
            if inTrackedFeatures is not None:
                features = inTrackedFeatures.trackedFeatures
                motion, vanishingPt = cameraEstimator.estimate_motion(featureDrawer.trackedFeaturesPath)
                featureDrawer.trackFeaturePath(features)
                featureDrawer.drawFeatures(leftFrame, vanishingPt, motion)
                cv2.imshow(left_window_name, leftFrame)
                print("Motions:", motion)

        # 4. MobileNet-SSD Object Detection (using right mono processed image)
        inRight = qRight.tryGet()
        if inRight is not None:
            rightFrame = inRight.getCvFrame()
            inDet = qNN.tryGet()
            detections = []
            if inDet is not None:
                detections = inDet.detections
            for detection in detections:
                bbox = frameNorm(rightFrame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.putText(rightFrame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
                cv2.putText(rightFrame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
                cv2.rectangle(rightFrame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.imshow("right", rightFrame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
