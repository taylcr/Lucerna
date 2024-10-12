import cv2
import time  # Import the time module to use sleep function

def test_camera_index(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Camera at index {index} is not available")
        return
    print(f"Testing camera at index {index}...")

    start_time = time.time()  # Record the start time

    # Loop to display the camera feed for 2 seconds
    while time.time() - start_time < 2:  # Run for 2 seconds
        ret, frame = cap.read()
        if ret:
            window_name = f"Camera Index {index}"
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
                break
        else:
            print(f"Failed to grab frame from camera at index {index}")
            break

    cv2.destroyWindow(window_name)
    cap.release()

# Test the default camera index 0
test_camera_index(0)
time.sleep(2)  # Wait for 2 seconds

# Test camera indices from 1 to 4
for i in range(1, 5):  # Correct range syntax
    test_camera_index(i)
    time.sleep(2)  # Wait for 2 seconds before moving on to the next camera index
