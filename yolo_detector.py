import cv2
import torch
import tkinter as tk
from PIL import Image, ImageTk

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

def detect_and_describe(feed=0, window=None, label=None, text_var=None, target_var=None):
    cap = cv2.VideoCapture(feed)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Update target object from entry
        target_object = target_var.get().strip().lower()

        # Perform detection
        results = model(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for displaying

        # Draw bounding boxes and label detected objects
        detected_objects = []
        for i, (x1, y1, x2, y2, conf, cls) in enumerate(results.xyxy[0]):
            label_text = results.names[int(cls)].lower()
            if label_text == target_object:
                color = (0, 255, 0)  # Green for target object
            else:
                color = (255, 0, 0)  # Red for others
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'{label_text} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detected_objects.append(label_text)

        # Convert the frame for Tkinter
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk  # Keep a reference
        label.configure(image=imgtk)

        # Update detected objects list
        text_var.set("Detected objects from left to right: " + ", ".join(detected_objects))
        window.update_idletasks()
        window.update()

def main():
    root = tk.Tk()
    root.title("YOLO Object Detection")

    target_var = tk.StringVar()  # Variable to hold the target object
    text_var = tk.StringVar(value="Detected objects will appear here.")  # Display detected objects

    # GUI Layout
    text_label = tk.Label(root, textvariable=text_var)
    text_label.pack()

    image_label = tk.Label(root)
    image_label.pack()

    entry = tk.Entry(root, textvariable=target_var)
    entry.pack()

    set_target_button = tk.Button(root, text="Set Target Object", command=lambda: target_var.set(entry.get().strip().lower()))
    set_target_button.pack()

    # Start the detection function
    root.after(0, detect_and_describe, 0, root, image_label, text_var, target_var)

    root.mainloop()

if __name__ == "__main__":
    main()
