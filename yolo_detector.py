import cv2
import torch
import tkinter as tk
from PIL import Image, ImageTk
import pyttsx3  # Text-to-Speech library


# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True, trust_repo=True)

# Initialize the Text-to-Speech engine
tts_engine = pyttsx3.init()

# Set confidence and NMS IoU thresholds to reduce jitter
model.conf = 0.4  # Increase confidence threshold to filter out weak detections
model.iou = 0.5   # Non-Maximum Suppression IoU threshold

# Dictionary to store the previous bounding boxes for smoothing
previous_boxes = {}

def smooth_bounding_boxes(current_box, previous_box, alpha=0.7):
    """ Smooths bounding boxes using a weighted average for smooth transitions """
    if previous_box is None:
        return current_box
    smoothed_box = [
        int(alpha * current + (1 - alpha) * previous)
        for current, previous in zip(current_box, previous_box)
    ]
    return smoothed_box

def detect_and_describe(feed, window, label, text_var, target_var, cap):
    global previous_boxes
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return

    target_object = target_var.get().strip().lower()

    # Perform detection
    results = model(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for displaying

    detected_objects = []
    for i, (x1, y1, x2, y2, conf, cls) in enumerate(results.xyxy[0]):
        label_text = results.names[int(cls)].lower()
        current_box = [int(x1), int(y1), int(x2), int(y2)]

        # If the object was detected in the previous frame, smooth the bounding box
        if label_text in previous_boxes:
            current_box = smooth_bounding_boxes(current_box, previous_boxes[label_text])

        # Store the current bounding box for the next frame
        previous_boxes[label_text] = current_box

        # Color based on whether it's the target object or not
        color = (0, 255, 0) if label_text == target_object else (255, 0, 0)

        # Draw smoothed bounding box
        cv2.rectangle(frame, (current_box[0], current_box[1]), (current_box[2], current_box[3]), color, 2)
        cv2.putText(frame, f'{label_text} {conf:.2f}', (current_box[0], current_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        detected_objects.append(label_text)

    # Display the image in Tkinter GUI
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    # Update detected objects in the Tkinter label
    text_var.set("Detected objects from left to right: " + ", ".join(detected_objects))
    window.after(10, detect_and_describe, feed, window, label, text_var, target_var, cap)

def swap_camera(current_camera, cap, window, image_label, text_var, target_var):
    # Release the current camera
    cap.release()
    
    # Switch to the next camera (0 to 1, or 1 to 0)
    new_camera_index = 1 if current_camera.get() == 0 else 0
    current_camera.set(new_camera_index)

    # Reinitialize VideoCapture with the new camera index
    cap = cv2.VideoCapture(new_camera_index)
    print(f"Switched to camera index {new_camera_index}")

    # Restart detection with the new camera
    window.after(0, detect_and_describe, new_camera_index, window, image_label, text_var, target_var, cap)

def speak_description(text_var):
    """ Function to speak the detected objects using TTS """
    description = text_var.get()
    tts_engine.say(description)  # Pass the description to the speech engine
    tts_engine.runAndWait()  # Ensure it completes speaking

def main():
    root = tk.Tk()
    root.title("YOLO Object Detection with TTS")

    target_var = tk.StringVar()
    text_var = tk.StringVar(value="Detected objects will be listed here.")
    current_camera = tk.IntVar(value=0)  # Default camera index is 0

    text_label = tk.Label(root, textvariable=text_var)
    text_label.pack()

    image_label = tk.Label(root)
    image_label.pack()

    entry = tk.Entry(root, textvariable=target_var)
    entry.pack()

    # Create buttons for setting target object, swapping camera, and speaking description
    set_target_button = tk.Button(root, text="Set Target Object", command=lambda: target_var.set(entry.get().strip().lower()))
    set_target_button.pack()

    swap_camera_button = tk.Button(root, text="Swap Camera", command=lambda: swap_camera(current_camera, cap, root, image_label, text_var, target_var))
    swap_camera_button.pack()

    # Button to trigger speech output
    speak_button = tk.Button(root, text="Speak Description", command=lambda: speak_description(text_var))
    speak_button.pack()

    # Capture from the default camera (index 0)
    cap = cv2.VideoCapture(current_camera.get())

    # Start the detection function with the default camera
    root.after(0, detect_and_describe, current_camera.get(), root, image_label, text_var, target_var, cap)

    root.mainloop()

if __name__ == "__main__":
    main()
