import cv2
import torch
import numpy as np
import pyttsx3         # For text-to-speech
import threading       # To run TTS non-blocking
from sort import Sort  # Ensure you have the SORT tracker installed
import pygame          # For audio feedback
import depthai as dai  # For OAK-D Lite

# Initialize pygame mixer
pygame.mixer.init()

# Load a smaller (optimized) YOLOv5 model for speed.
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
model.conf = 0.4  # Confidence threshold
model.iou = 0.5   # NMS IoU threshold
model.eval()

# Initialize the SORT tracker
tracker = Sort()

# Global variables for object tracking and control
tracked_id = None
id_buffer = ""       # Buffer for manual ID entry
selection_mode = False
selected_object_idx = 0
object_id_map = {}
next_id = 1

# Distance and guidance thresholds
CENTER_TOLERANCE = 50
CLOSE_ENOUGH_THRESHOLD = 20000  # (for area comparison; adjust as needed)
object_centered_once = False    # To say "centered" only once

# Text-to-speech engine
tts_engine = pyttsx3.init()

def speak_async(text):
    """Run TTS in a separate thread."""
    threading.Thread(target=lambda: (tts_engine.say(text), tts_engine.runAndWait())).start()

def get_unique_id():
    global next_id
    unique_id = next_id
    next_id += 1
    return unique_id

def assign_ids(tracked_objects):
    """Assign unique (simplified) IDs to tracked objects."""
    global object_id_map
    for obj in tracked_objects:
        tracker_id = int(obj[-1])
        if tracker_id not in object_id_map:
            object_id_map[tracker_id] = get_unique_id()
        obj[-1] = object_id_map[tracker_id]

def play_sound(sound_file):
    """Play the specified sound file."""
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()

def guidance_feedback(object_box, frame_center, current_area, previous_area):
    """
    Provide guidance feedback based on the object's bounding box and frame center.
    (This function can be extended to also use depth if desired.)
    """
    global object_centered_once
    x1, y1, x2, y2 = object_box
    x_frame, y_frame = frame_center

    if x1 <= x_frame <= x2 and y1 <= y_frame <= y2:
        if not object_centered_once:
            speak_async("centered")
            print("centered")
            object_centered_once = True

        if current_area >= CLOSE_ENOUGH_THRESHOLD:
            play_sound("end.mp3")
        elif previous_area is not None and current_area > previous_area:
            play_sound("step.mp3")
    else:
        object_centered_once = False
        directions = []
        if x_frame < x1:
            directions.append("right")
        elif x_frame > x2:
            directions.append("left")
        if y_frame < y1:
            directions.append("down")
        elif y_frame > y2:
            directions.append("up")
        if directions:
            direction_text = ", ".join(directions)
            speak_async(direction_text)
            print(direction_text)

def describe_objects(detections, class_names):
    """Describe detected objects from left to right."""
    descriptions = []
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, obj_id = map(int, detection[:5])
        descriptions.append(f"{class_names[i]} ID {obj_id}")
    if descriptions:
        description_text = " | ".join(descriptions)
        speak_async(description_text)
        print(description_text)

def cycle_objects(key, tracked_objects, class_names):
    """Cycle through objects when in selection mode."""
    global selected_object_idx
    if tracked_objects.size > 0 and selection_mode:
        num_objects = len(tracked_objects)
        selected_object_idx = (selected_object_idx + (1 if key == ord('l') else -1)) % num_objects
        x1, y1, x2, y2, obj_id = map(int, tracked_objects[selected_object_idx][:5])
        object_name = class_names[selected_object_idx] if selected_object_idx < len(class_names) else "object"
        speak_async(f"{object_name} with ID {obj_id}")
        print(f"Selected: {object_name} ID: {obj_id}")

def reset_tracking():
    """Reset tracking and selection states."""
    global tracked_id, id_buffer, selection_mode, selected_object_idx, object_centered_once
    tracked_id = None
    id_buffer = ""
    selection_mode = False
    selected_object_idx = 0
    object_centered_once = False
    print("Tracking and selection reset.")

###############################################################################
# Build the DepthAI Pipeline for OAK-D Lite (RGB + Depth)
###############################################################################

pipeline = dai.Pipeline()

# Color camera node (using CAM_A)
camRgb = pipeline.createColorCamera()
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# Use a smaller preview size for faster inference (e.g., 320x240)
camRgb.setPreviewSize(320, 240)
camRgb.setInterleaved(False)

# Mono cameras for depth (using CAM_B and CAM_C)
monoLeft = pipeline.createMonoCamera()
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
# For OAK-D Lite, use a supported resolution (e.g., 400p)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

monoRight = pipeline.createMonoCamera()
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# StereoDepth node
stereo = pipeline.createStereoDepth()
stereo.setConfidenceThreshold(50)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(True)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# XLinkOut for the RGB preview
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

# XLinkOut for the aligned depth output
xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth_aligned")
stereo.depth.link(xoutDepth.input)

###############################################################################
# Main loop: Run YOLOv5, SORT tracking, and overlay depth information
###############################################################################
def main():
    global tracked_id, id_buffer, selection_mode, selected_object_idx, object_id_map

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDepth = device.getOutputQueue(name="depth_aligned", maxSize=4, blocking=False)
        print("Starting OAK-D Lite stream...")

        previous_area = None

        while True:
            inRgb = qRgb.get()
            inDepth = qDepth.get()
            if inRgb is None or inDepth is None:
                continue

            frame = inRgb.getCvFrame()         # 320x240 color frame
            depth_frame = inDepth.getFrame()     # Depth frame (typically 16-bit, in millimeters)

            # Get dimensions of the color frame and depth frame
            color_h, color_w = frame.shape[:2]
            depth_h, depth_w = depth_frame.shape[:2]
            # Compute scale factors in case resolutions differ.
            scale_x = depth_w / color_w
            scale_y = depth_h / color_h

            # Draw crosshair at the center of the color frame.
            frame_center = (color_w // 2, color_h // 2)
            cv2.drawMarker(frame, frame_center, (255, 255, 255), cv2.MARKER_CROSS, thickness=2)

            # Run YOLOv5 detection on the color frame.
            with torch.no_grad():
                results = model(frame)
            dets = []
            class_names = []
            # Each detection: [x1, y1, x2, y2, conf, cls]
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                dets.append([*xyxy, conf])
                class_names.append(results.names[int(cls)])
            dets = np.array(dets)

            # If no detections, show the frame and continue.
            if dets.size == 0:
                cv2.imshow('OAK-D Lite: YOLO & Depth', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Update the SORT tracker and assign simplified IDs.
            tracked_objects = tracker.update(dets)
            assign_ids(tracked_objects)

            key = cv2.waitKey(1) & 0xFF

            for i, obj in enumerate(tracked_objects):
                x1, y1, x2, y2, obj_id = map(int, obj[:5])
                # Compute the center of the bounding box in the color image.
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                # Map the center point to the depth frame coordinates.
                cx_depth = int(cx * scale_x)
                cy_depth = int(cy * scale_y)
                if 0 <= cy_depth < depth_h and 0 <= cx_depth < depth_w:
                    depth_val = depth_frame[cy_depth, cx_depth]
                else:
                    depth_val = 0
                distance_m = depth_val / 1000.0  # Convert mm to meters

                object_name = class_names[i] if i < len(class_names) else "object"
                label_text = f'{object_name} ID: {obj_id} {distance_m:.2f}m'

                if selection_mode and i == selected_object_idx:
                    color = (255, 0, 0)  # Blue for selected object
                elif tracked_id is not None and obj_id == tracked_id:
                    color = (0, 0, 255)  # Red for actively tracked object
                    object_box = (x1, y1, x2, y2)
                    current_area = (x2 - x1) * (y2 - y1)
                    guidance_feedback(object_box, frame_center, current_area, previous_area)
                    previous_area = current_area
                else:
                    color = (0, 255, 0)  # Green for others

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('OAK-D Lite: YOLO & Depth', frame)

            if key == ord('q'):
                break
            elif key == ord('v'):
                describe_objects(tracked_objects, class_names)
            elif key == ord('t'):
                selection_mode = True
                selected_object_idx = 0
                if tracked_objects.size > 0:
                    x1, y1, x2, y2, obj_id = map(int, tracked_objects[selected_object_idx][:5])
                    speak_async(f"Selection mode. {class_names[selected_object_idx]} with ID {obj_id}")
                    print(f"Selection mode: {class_names[selected_object_idx]} ID: {obj_id}")
            elif key == ord('r'):
                reset_tracking()
            elif ord('0') <= key <= ord('9'):
                id_buffer += chr(key)
                print(f"Building ID: {id_buffer}")
            elif key == 13:  # Enter key
                try:
                    tracked_id = int(id_buffer if not selection_mode else tracked_objects[selected_object_idx][-1])
                    id_buffer = ""
                    selection_mode = False
                    speak_async(f"Tracking object ID: {tracked_id}")
                    print(f"Tracking object ID: {tracked_id}")
                except ValueError:
                    print("Invalid ID entered")
                    id_buffer = ""
            elif key in {ord('j'), ord('l')} and selection_mode:
                cycle_objects(key, tracked_objects, class_names)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
