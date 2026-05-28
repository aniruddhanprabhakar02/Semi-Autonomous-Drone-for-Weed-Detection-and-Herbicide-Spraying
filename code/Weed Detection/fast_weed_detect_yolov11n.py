import cv2
from ultralytics import YOLO
import cvzone
import numpy as np
import time
import threading

# ---------------------------------------------------------
# Load YOLOv11n weed detection model
# ---------------------------------------------------------
model = YOLO("weed11n.pt")

# Load class list
with open("weed_classes.txt", "r") as f:
    class_list = f.read().split("\n")


# ---------------------------------------------------------
# Threaded Webcam Class (Raspberry Pi FIXED)
# ---------------------------------------------------------
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)  # FIX for Pi USB camera
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        time.sleep(0.2)  # allow camera to warm up

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


# ---------------------------------------------------------
# Start the camera
# ---------------------------------------------------------
vs = VideoStream(0)
prev_time = 0

while True:
    frame = vs.read()

    if frame is None:
        print("No frame captured!")
        continue

    frame = cv2.flip(frame, 1)

    # FPS calculation
    now = time.time()
    fps = 1 / (now - prev_time) if prev_time else 0
    prev_time = now

    # YOLO inference
    results = model(frame, imgsz=416, conf=0.40, verbose=False)
    boxes = results[0].boxes

    # Draw boxes
    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cls = int(box.cls[0])
            cname = class_list[cls]
            color = (0, 0, 255) if cname == "weed" else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cvzone.putTextRect(frame, f"{cname}", (x1, y1 - 5),
                               scale=1, thickness=1)

    # Show FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # Display video
    cv2.imshow("YOLOv11n Weed Detection - Raspberry Pi 4B", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
