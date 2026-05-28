import cv2
import numpy as np
import time
import threading
import tflite_runtime.interpreter as tflite
import cvzone

# ---------------------------------------------------------
# Load class names
# ---------------------------------------------------------
with open("weed_classes.txt") as f:
    class_list = f.read().strip().split("\n")

# ---------------------------------------------------------
# Load TFLite YOLO model
# ---------------------------------------------------------
interpreter = tflite.Interpreter(model_path="best_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]["shape"][1]
input_width = input_details[0]["shape"][2]

# ---------------------------------------------------------
# Threaded USB camera
# ---------------------------------------------------------
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        time.sleep(0.2)
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
# YOLOv11n TFLite inference
# ---------------------------------------------------------
def detect_objects(frame):
    img_resized = cv2.resize(frame, (input_width, input_height))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_resized)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]

    detections = []

    for det in output:
        vals = det.tolist()

        # bounding box (first 4 numbers)
        x1, y1, x2, y2 = map(int, vals[:4])

        # confidence (5th value)
        score = float(vals[4])
        if score < 0.40:
            continue

        # class id (last value)
        cls = int(vals[-1])

        detections.append([x1, y1, x2, y2, score, cls])

    return detections

# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------
vs = VideoStream(0)
prev_time = 0

while True:
    frame = vs.read()
    if frame is None:
        continue

    now = time.time()
    fps = 1 / (now - prev_time) if prev_time else 0
    prev_time = now

    detections = detect_objects(frame)

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cname = class_list[cls]

        color = (0, 0, 255) if cname.lower() == "weed" else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cvzone.putTextRect(frame, f"{cname} {conf:.2f}",
                           (x1, y1 - 5), scale=1, thickness=1)

    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2)

    cv2.imshow("YOLOv11n TFLite - Raspberry Pi 4B", frame)

    if cv2.waitKey(1) == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()
