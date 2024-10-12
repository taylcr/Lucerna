import cv2
import torch
import tkinter as tk
from PIL import Image, ImageTk

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

def detect_and_describe(feed, window, label, text_var, target_var, cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        target_object = target_var.get().strip().lower()

        # Perform detection
        results = model(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for displaying

        # Draw bounding boxes and label detected objects
        detected_objects = []
        for i, (x1, y1, x2, y2, conf, cls) in enumerate(results.xyxy[0]):
            label_text = results.names[int(cls)].lower()
            color = (0, 255, 0) if label_text == target_object else (255, 0, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'{label_text} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detected_objects.append(label_text)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        text_var.set("Detected objects from left to right: " + ", ".join(detected_objects))
        window.update_idletasks()
        window.update()

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

def main():
    root = tk.Tk()
    root.title("YOLO Object Detection")

    target_var = tk.StringVar()
    text_var = tk.StringVar(value="Detected objects will be listed here.")
    current_camera = tk.IntVar(value=0)  # Default camera index is 0

    text_label = tk.Label(root, textvariable=text_var)
    text_label.pack()

    image_label = tk.Label(root)
    image_label.pack()

    entry = tk.Entry(root, textvariable=target_var)
    entry.pack()

    # Create buttons for setting target object and swapping camera
    set_target_button = tk.Button(root, text="Set Target Object", command=lambda: target_var.set(entry.get().strip().lower()))
    set_target_button.pack()

    # Capture from the default camera (index 0)
    cap = cv2.VideoCapture(current_camera.get())

    swap_camera_button = tk.Button(root, text="Swap Camera", command=lambda: swap_camera(current_camera, cap, root, image_label, text_var, target_var))
    swap_camera_button.pack()

    # Start the detection function with the default camera
    root.after(0, detect_and_describe, current_camera.get(), root, image_label, text_var, target_var, cap)

    root.mainloop()

if __name__ == "__main__":
    main()
