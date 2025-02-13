import cv2
import torch
import numpy as np
import pyttsx3  # For text-to-speech
import threading  # To make TTS non-blocking
from sort import Sort
import pygame  # For audio feedback

# Initialize pygame mixer
pygame.mixer.init()

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True, trust_repo=True)

# Set confidence and NMS IoU thresholds
model.conf = 0.4
model.iou = 0.5

# Initialize the SORT tracker
tracker = Sort()

# Variables for object tracking and control
tracked_id = None
id_buffer = ""  # Buffer for ID entry
selection_mode = False
selected_object_idx = 0

# Text-to-speech engine
tts_engine = pyttsx3.init()

# Camera toggle
camera_index = 0

# Distance threshold and tolerance for "centered" detection
CENTER_TOLERANCE = 50
CLOSE_ENOUGH_THRESHOLD = 20000
object_centered_once = False  # To say "centered" only once

# Object ID mapping
object_id_map = {}
next_id = 1

# Function to run TTS in a separate thread
def speak_async(text):
    threading.Thread(target=lambda: tts_engine.say(text) or tts_engine.runAndWait()).start()

def get_unique_id():
    """Get a unique ID and increment for the next detection."""
    global next_id
    unique_id = next_id
    next_id += 1
    return unique_id

def assign_ids(tracked_objects):
    """Assign unique, simplified IDs to each detected object."""
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
    """Provide guidance and proximity feedback when the crosshair is within the bounding box."""
    global object_centered_once
    x1, y1, x2, y2 = object_box  # Bounding box coordinates
    x_frame, y_frame = frame_center

    # Check if the frame center (crosshair) is within the bounding box
    if x1 <= x_frame <= x2 and y1 <= y_frame <= y2:
        if not object_centered_once:
            speak_async("centered")  # Non-blocking TTS
            print("centered")
            object_centered_once = True  # Mark as centered

        # Proximity feedback only when "centered"
        if current_area >= CLOSE_ENOUGH_THRESHOLD:
            play_sound("end.mp3")
        elif previous_area is not None and current_area > previous_area:
            play_sound("step.mp3")
    else:
        # If not centered, reset the centered flag for the next time it aligns
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
            speak_async(direction_text)  # Non-blocking TTS
            print(direction_text)

def describe_objects(detections, class_names):
    """Describe detected objects from left to right."""
    descriptions = []
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, obj_id = map(int, detection[:5])
        descriptions.append(f"{class_names[i]} ID {obj_id}")
    if descriptions:
        description_text = " | ".join(descriptions)
        speak_async(description_text)  # Non-blocking TTS
        print(description_text)

def cycle_objects(key, tracked_objects, class_names):
    """Cycle through objects with 'j' and 'l' in selection mode."""
    global selected_object_idx
    if tracked_objects.size > 0 and selection_mode:
        num_objects = len(tracked_objects)
        selected_object_idx = (selected_object_idx + (1 if key == ord('l') else -1)) % num_objects

        # Announce the selected object
        x1, y1, x2, y2, obj_id = map(int, tracked_objects[selected_object_idx][:5])
        object_name = class_names[selected_object_idx]
        speak_async(f"{object_name} with ID {obj_id}")  # Non-blocking TTS
        print(f"Selected: {object_name} ID: {obj_id}")

def reset_tracking():
    """Resets all tracking and selection states."""
    global tracked_id, id_buffer, selection_mode, selected_object_idx, object_centered_once
    tracked_id = None
    id_buffer = ""
    selection_mode = False
    selected_object_idx = 0
    object_centered_once = False  # Reset centered state on reset
    print("Tracking and selection reset.")

def switch_camera(cap):
    """Switch camera between indices."""
    global camera_index
    cap.release()
    camera_index = 1 if camera_index == 0 else 0
    return cv2.VideoCapture(camera_index)

def main():
    global tracked_id, id_buffer, selection_mode, selected_object_idx, object_id_map, camera_index

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    previous_area = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_height, frame_width = frame.shape[:2]
        frame_center = (frame_width // 2, frame_height // 2)
        cv2.drawMarker(frame, frame_center, (255, 255, 255), markerType=cv2.MARKER_CROSS, thickness=2)

        results = model(frame)
        dets = []
        class_names = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            dets.append([*xyxy, conf])
            class_names.append(results.names[int(cls)])

        dets = np.array(dets)
        if dets.size == 0:
            cv2.imshow('YOLO Object Detection and SORT Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        tracked_objects = tracker.update(dets)
        assign_ids(tracked_objects)

        key = cv2.waitKey(1) & 0xFF
        for i, obj in enumerate(tracked_objects):
            x1, y1, x2, y2, obj_id = map(int, obj[:5])
            object_name = class_names[i]
            label_text = f'{object_name} ID: {obj_id}'

            if selection_mode and i == selected_object_idx:
                color = (255, 0, 0)
            elif tracked_id is not None and obj_id == tracked_id:
                color = (0, 0, 255)
                object_box = (x1, y1, x2, y2)
                current_area = (x2 - x1) * (y2 - y1)
                guidance_feedback(object_box, frame_center, current_area, previous_area)
                previous_area = current_area
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('YOLO Object Detection and SORT Tracking', frame)

        if key == ord('q'):
            break
        elif key == ord('c'):
            cap = switch_camera(cap)
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
        elif key == 13:
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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
