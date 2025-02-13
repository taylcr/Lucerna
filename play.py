#!/usr/bin/env python3
"""
OAK-D Lite Navigation & Guidance System for the Visually Impaired

This meta-app uses the OAK-D Lite’s streams to:
  • Run YOLO‑v5 object detection on a 640×480 RGB preview.
  • Map detections into an aligned depth frame (to compute distances).
  • Provide “go‑to” guidance (via TTS and sound cues) when an object is selected.
  • Display a colorized depth (disparity) map.
  • Show collision avoidance information via a spatial location calculator.
  • Optionally display feature motion estimation from the left mono camera.

Extra windows ("Depth Map", "Collision Avoidance", "Feature Motion") are shown only if devmode is True.
  
Key Controls in the Main window:
  s : Start tracking the first detected object.
  n : Cycle through detections.
  c : Cancel tracking.
  t : Describe (from left to right) every object currently detected.
  q : Quit.
  
Additional keys (e.g. t, j, l) can be added as desired.

**Set `devmode` to True to open the extra windows and use a larger Main window;
if False, only the Main window (640×480) is shown.**
"""

import cv2
import torch
import numpy as np
import depthai as dai
import pyttsx3
import threading
import time
import pygame
from math import sqrt
from sort import Sort  # Ensure you have SORT installed
from collections import deque

# ------------- User Configuration -------------
devmode = True  # Set True to enable extra windows and larger Main window
# ------------- End User Configuration -------------

# -------------------------------
# FEATURE MOTION ESTIMATION CLASSES
# -------------------------------
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
        if num_features == 0:
            return most_prominent_motion, vanishing_point
        for path in feature_paths.values():
            if len(path) >= 2:
                src = np.array([path[-2].x, path[-2].y])
                dst = np.array([path[-1].x, path[-1].y])
                avg_flow += (dst - src)
                motion_vector = dst + (dst - src)
                vanishing_point += motion_vector
                rotation = np.arctan2(dst[1]-src[1], dst[0]-src[0])
                total_rotation += rotation
        avg_flow /= num_features
        avg_rotation = total_rotation / num_features
        vanishing_point /= num_features
        avg_flow = (self.filter_weight * self.last_avg_flow + (1 - self.filter_weight) * avg_flow)
        self.last_avg_flow = avg_flow
        flow_magnitude = np.linalg.norm(avg_flow)
        rotation_magnitude = abs(avg_rotation)
        if flow_magnitude > self.motion_threshold:
            if abs(avg_flow[0]) > abs(avg_flow[1]):
                most_prominent_motion = 'Right' if avg_flow[0] < 0 else 'Left'
            else:
                most_prominent_motion = 'Down' if avg_flow[1] < 0 else 'Up'
            max_magnitude = flow_magnitude
        if rotation_magnitude > self.rotation_threshold and rotation_magnitude > max_magnitude:
            most_prominent_motion = 'Rotating'
        return most_prominent_motion, vanishing_point

class FeatureTrackerDrawer:
    lineColor = (200, 0, 200)
    pointColor = (0, 0, 255)
    vanishingPointColor = (255, 0, 255)
    circleRadius = 2
    trackedFeaturesPathLength = 30

    def __init__(self, windowName):
        self.windowName = windowName
        cv2.namedWindow(windowName)
        self.trackedIDs = set()
        self.trackedFeaturesPath = dict()

    def trackFeaturePath(self, features):
        newTrackedIDs = set()
        for feature in features:
            currentID = feature.id
            newTrackedIDs.add(currentID)
            if currentID not in self.trackedFeaturesPath:
                self.trackedFeaturesPath[currentID] = deque()
            path = self.trackedFeaturesPath[currentID]
            path.append(feature.position)
            while len(path) > FeatureTrackerDrawer.trackedFeaturesPathLength:
                path.popleft()
            self.trackedFeaturesPath[currentID] = path
        for oldId in list(self.trackedFeaturesPath.keys()):
            if oldId not in newTrackedIDs:
                self.trackedFeaturesPath.pop(oldId)
        self.trackedIDs = newTrackedIDs

    def drawFeatures(self, img, vanishing_point=None, prominent_motion=None):
        point_color = self.pointColor
        if prominent_motion in {"Up", "Down", "Left", "Right", "Rotating"}:
            mapping = {"Up": (0, 255, 255), "Down": (0, 255, 0),
                       "Left": (255, 0, 0), "Right": (0, 0, 255), "Rotating": (255, 255, 0)}
            point_color = mapping.get(prominent_motion, self.pointColor)
        for path in self.trackedFeaturesPath.values():
            for j in range(len(path)-1):
                src = (int(path[j].x), int(path[j].y))
                dst = (int(path[j+1].x), int(path[j+1].y))
                cv2.line(img, src, dst, point_color, 1, cv2.LINE_AA)
            if len(path) > 0:
                pt = path[-1]
                cv2.circle(img, (int(pt.x), int(pt.y)), self.circleRadius, point_color, -1, cv2.LINE_AA)
        if prominent_motion:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, prominent_motion, (10,30), font, 1, point_color, 2, cv2.LINE_AA)
        if vanishing_point is not None:
            cv2.circle(img, (int(vanishing_point[0]), int(vanishing_point[1])), self.circleRadius, self.vanishingPointColor, -1, cv2.LINE_AA)

# -------------------------------
# AUDIO & TTS Setup
# -------------------------------
tts_engine = pyttsx3.init()
def speak_async(text):
    threading.Thread(target=lambda: (tts_engine.say(text), tts_engine.runAndWait())).start()

pygame.mixer.init()
def play_sound(filename):
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

# -------------------------------
# Object Detection & Tracking Setup
# -------------------------------
# Use YOLOv5s for speed.
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
model.conf = 0.4
model.iou = 0.5
model.eval()
tracker = Sort()  # SORT tracker for YOLO detections

# For object selection/tracking in Main window:
selected_object = None   # index of currently selected detection
tracking_active = False

# -------------------------------
# Collision Avoidance Parameters
# -------------------------------
OBSTACLE_THRESHOLD = 0.8  # meters
last_collision_warn_time = 0
COLLISION_WARN_DELAY = 3  # seconds between warnings

# -------------------------------
# Build Combined DepthAI Pipeline
# -------------------------------
pipeline = dai.Pipeline()

# Color camera (CAM_A): used for YOLO detection; set preview to 640×480.
camRgb = pipeline.createColorCamera()
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setPreviewSize(640, 480)
camRgb.setInterleaved(False)

# Mono cameras for stereo depth (CAM_B and CAM_C)
monoLeft = pipeline.createMonoCamera()
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight = pipeline.createMonoCamera()
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Stereo Depth node: aligned to CAM_A
stereo = pipeline.createStereoDepth()
stereo.setConfidenceThreshold(50)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(True)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Colormap for disparity: use ImageManip.
colormap_manip = pipeline.createImageManip()
colormap_manip.initialConfig.setColormap(dai.Colormap.STEREO_TURBO, stereo.initialConfig.getMaxDisparity())
colormap_manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
colormap_manip.setMaxOutputFrameSize(4194304)  # Set to 4MB to avoid output frame size errors.
stereo.disparity.link(colormap_manip.inputImage)

# Spatial Location Calculator for collision avoidance.
slc = pipeline.createSpatialLocationCalculator()
for x in range(15):
    for y in range(9):
        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 200
        config.depthThresholds.upperThreshold = 10000
        config.roi = dai.Rect(dai.Point2f((x+0.5)*0.0625, (y+0.5)*0.1),
                              dai.Point2f((x+1.5)*0.0625, (y+1.5)*0.1))
        config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
        slc.initialConfig.addROI(config)
stereo.depth.link(slc.inputDepth)

# Feature Tracker for left mono camera (for feature motion estimation).
featureTracker = pipeline.createFeatureTracker()
featureTracker.setHardwareResources(2, 2)
monoLeft.out.link(featureTracker.inputImage)

# -------------------------------
# XLinkOut Nodes for Host Streaming
# -------------------------------
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

xoutDisparity = pipeline.createXLinkOut()
xoutDisparity.setStreamName("disparity")
colormap_manip.out.link(xoutDisparity.input)

xoutSlc = pipeline.createXLinkOut()
xoutSlc.setStreamName("slc")
slc.out.link(xoutSlc.input)

# Only create feature tracker XLinkOut nodes if devmode is True.
if devmode:
    xoutFeat = pipeline.createXLinkOut()
    xoutFeat.setStreamName("trackedFeaturesLeft")
    featureTracker.outputFeatures.link(xoutFeat.input)
    
    xoutFeatPass = pipeline.createXLinkOut()
    xoutFeatPass.setStreamName("passthroughLeft")
    featureTracker.passthroughInputImage.link(xoutFeatPass.input)

# -------------------------------
# Host Side Processing: Open Windows & Process Streams
# -------------------------------
def main_app():
    global selected_object, tracking_active, last_collision_warn_time

    # Set Main window size based on devmode:
    main_window_size = (1280, 960) if devmode else (640, 480)
    cv2.namedWindow("Main", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Main", *main_window_size)
    
    # Create extra windows only if devmode is enabled.
    if devmode:
        cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth Map", 640, 480)
        cv2.namedWindow("Collision Avoidance", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Collision Avoidance", 640, 480)
        cv2.namedWindow("Feature Motion", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Feature Motion", 640, 480)

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        qDisparity = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
        qSlc = device.getOutputQueue(name="slc", maxSize=4, blocking=False)
        
        if devmode:
            qFeat = device.getOutputQueue(name="trackedFeaturesLeft", maxSize=4, blocking=False)
            qFeatPass = device.getOutputQueue(name="passthroughLeft", maxSize=4, blocking=False)
        
        # For feature motion estimation:
        if devmode:
            featDrawer = FeatureTrackerDrawer("Feature Motion")
            cameraEstimator = CameraMotionEstimator(filter_weight=0.5, motion_threshold=0.03, rotation_threshold=0.05)

        while True:
            # -------- Main Window: YOLO Detection & Guidance --------
            inRgb = qRgb.get()
            inDepth = qDepth.get()
            if inRgb is None or inDepth is None:
                continue
            mainFrame = inRgb.getCvFrame()  # 640x480 color frame
            depthFrame = inDepth.getFrame()  # depth frame in mm

            # Compute scale factors between color and depth frames.
            color_h, color_w = mainFrame.shape[:2]
            depth_h, depth_w = depthFrame.shape[:2]
            scale_x = depth_w / color_w
            scale_y = depth_h / color_h

            frame_center = (color_w // 2, color_h // 2)
            cv2.drawMarker(mainFrame, frame_center, (255,255,255), cv2.MARKER_CROSS, thickness=2)

            # Run YOLO detection on mainFrame.
            with torch.no_grad():
                results = model(mainFrame)
            dets = []
            class_names = []
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                dets.append([*xyxy, conf])
                class_names.append(results.names[int(cls)])
            dets = np.array(dets)
            if dets.size > 0:
                tracked_objects = tracker.update(dets)
                for i, obj in enumerate(tracked_objects):
                    x1, y1, x2, y2, obj_id = map(int, obj[:5])
                    cx = int((x1+x2)/2)
                    cy = int((y1+y2)/2)
                    cx_d = int(cx * scale_x)
                    cy_d = int(cy * scale_y)
                    if 0 <= cx_d < depth_w and 0 <= cy_d < depth_h:
                        d_val = depthFrame[cy_d, cx_d]
                    else:
                        d_val = 0
                    distance_m = d_val / 1000.0
                    label_text = f"{class_names[i]} {distance_m:.2f}m"
                    color = (0,255,0)
                    if tracking_active and selected_object is not None and i == selected_object:
                        color = (0,0,255)
                        diff_x = cx - frame_center[0]
                        diff_y = cy - frame_center[1]
                        instr = []
                        if abs(diff_x) > 20:
                            instr.append("left" if diff_x < 0 else "right")
                        if abs(diff_y) > 20:
                            instr.append("up" if diff_y < 0 else "down")
                        instr.append(f"{distance_m:.2f}m away")
                        feedback = ", ".join(instr)
                        cv2.putText(mainFrame, "TRACK: "+feedback, (x1, y1-30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        if time.time() - last_collision_warn_time > 2:
                            speak_async(feedback)
                            last_collision_warn_time = time.time()
                    cv2.rectangle(mainFrame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(mainFrame, label_text, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # --------- NEW: Describe objects from left to right when "t" is pressed ---------
                key = cv2.waitKey(1) & 0xFF  # Check key here to allow immediate description
                if key == ord('t'):
                    # Sort tracked_objects (or detections) by x1 coordinate.
                    sorted_idx = np.argsort([int(obj[0]) for obj in tracked_objects])
                    description = []
                    for idx in sorted_idx:
                        x1, y1, x2, y2, obj_id = map(int, tracked_objects[idx][:5])
                        cx = int((x1+x2)/2)
                        cy = int((y1+y2)/2)
                        cx_d = int(cx * scale_x)
                        cy_d = int(cy * scale_y)
                        if 0 <= cx_d < depth_w and 0 <= cy_d < depth_h:
                            d_val = depthFrame[cy_d, cx_d]
                        else:
                            d_val = 0
                        distance_m = d_val / 1000.0
                        description.append(f"{class_names[idx]} at {distance_m:.2f}m")
                    if description:
                        speak_async(" ".join(description))
                        print("Described:", " ".join(description))
                else:
                    # Otherwise, process the key later
                    pass

            cv2.imshow("Main", mainFrame)

            # -------- Depth Map Window --------
            if devmode:
                inDisp = qDisparity.get()
                if inDisp is not None:
                    dispFrame = inDisp.getCvFrame()
                    cv2.imshow("Depth Map", dispFrame)

            # -------- Collision Avoidance Window --------
            if devmode:
                inSlc = qSlc.tryGet()  # non-blocking
                collFrame = mainFrame.copy()
                if inSlc is not None:
                    slc_data = inSlc.getSpatialLocations()
                    for data in slc_data:
                        roi = data.config.roi.denormalize(width=collFrame.shape[1], height=collFrame.shape[0])
                        xmin = int(roi.topLeft().x)
                        ymin = int(roi.topLeft().y)
                        xmax = int(roi.bottomRight().x)
                        ymax = int(roi.bottomRight().y)
                        coords = data.spatialCoordinates
                        dist = sqrt(coords.x**2 + coords.y**2 + coords.z**2)
                        if dist > 0 and dist < OBSTACLE_THRESHOLD * 1000:
                            cv2.rectangle(collFrame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                            cv2.putText(collFrame, f"{dist/1000:.1f}m", (xmin+5, ymin+20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                            if time.time() - last_collision_warn_time > COLLISION_WARN_DELAY:
                                play_sound("step.mp3")
                                last_collision_warn_time = time.time()
                cv2.imshow("Collision Avoidance", collFrame)

            # -------- Feature Motion Window --------
            if devmode:
                inFeatPass = qFeatPass.get()
                inFeat = qFeat.get()
                if inFeatPass is not None and inFeat is not None:
                    featFrame = inFeatPass.getCvFrame()  # grayscale
                    featFrameColor = cv2.cvtColor(featFrame, cv2.COLOR_GRAY2BGR)
                    tracked_features = inFeat.trackedFeatures
                    featDrawer.trackFeaturePath(tracked_features)
                    motion, vanish_pt = cameraEstimator.estimate_motion(featDrawer.trackedFeaturesPath)
                    featDrawer.drawFeatures(featFrameColor, vanish_pt, motion)
                    cv2.imshow("Feature Motion", featFrameColor)

            # -------- Handle Other Key Controls --------
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if dets.size > 0:
                    selected_object = 0
                    tracking_active = True
                    label = class_names[0]
                    cx = int((int(dets[0][0]) + int(dets[0][2]))/2)
                    cy = int((int(dets[0][1]) + int(dets[0][3]))/2)
                    cx_d = int(cx * scale_x)
                    cy_d = int(cy * scale_y)
                    if 0 <= cx_d < depth_w and 0 <= cy_d < depth_h:
                        d_val = depthFrame[cy_d, cx_d]
                    else:
                        d_val = 0
                    speak_async(f"Tracking {label} at {d_val/1000:.2f} meters.")
            elif key == ord('n'):
                if tracking_active and dets.size > 0:
                    selected_object = (selected_object + 1) % dets.shape[0]
                    label = class_names[int(selected_object)]
                    speak_async(f"Selected {label}.")
            elif key == ord('c'):
                tracking_active = False
                selected_object = None
                speak_async("Tracking canceled.")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    if devmode:
        from sys import argv  # if you want to pass arguments to feature motion
        featDrawer = FeatureTrackerDrawer("Feature Motion")
        cameraEstimator = CameraMotionEstimator(filter_weight=0.5, motion_threshold=0.03, rotation_threshold=0.05)
    main_app()
