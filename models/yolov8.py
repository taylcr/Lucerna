import socket
import cv2
import pickle
import struct
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.

# Set up socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '192.168.8.175'  # IP address of the server (remote computer)
port = 9999

client_socket.connect((host_ip, port))

data = b""
payload_size = struct.calcsize("Q")

while True:
    # Receive data from server
    while len(data) < payload_size:
        packet = client_socket.recv(4*1024)  # Buffer size
        if not packet:
            break
        data += packet

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    while len(data) < msg_size:
        data += client_socket.recv(4*1024)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Deserialize frame
    frame = pickle.loads(frame_data)

    # Run YOLOv8 on the frame
    results = model(frame)

    # Annotate frame with YOLOv8 results
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()
cv2.destroyAllWindows()
