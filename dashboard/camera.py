import cv2
import numpy as np
import datetime
import csv
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


class VideoCamera:
    def __init__(self):
        self.cap = None
        self.running = False   # ⭐ FLAG QUAN TRỌNG

        # ================= LOAD MODEL =================
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(
            current_dir, '..', 'models', 'mobilenet_tomato_finetune.h5'
        )
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)

        self.classes = ['unripe', 'semi-ripe', 'fully-ripe']
        self.img_size = (150, 150)

        # ================= LOGGING =================
        self.log_file = os.path.join(
            current_dir, '..', 'detection_logs.csv'
        )
        self.last_log_time = datetime.datetime.min
        self.log_interval = datetime.timedelta(seconds=2)

        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Class', 'Confidence'])

        print("Camera object initialized (NOT started)")

    # ================= CAMERA CONTROL =================
    def start(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            print("Camera started")

    def stop(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.running = False
        print("Camera stopped")

    def __del__(self):
        self.stop()

    # ================= MAIN FRAME FUNCTION =================
    def get_frame(self):
        # -------- CAMERA OFF --------
        if not self.running:
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(
                blank,
                "Camera Stopped",
                (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            _, jpeg = cv2.imencode('.jpg', blank)
            return jpeg.tobytes()

        # -------- READ FRAME --------
        ret, frame = self.cap.read()
        if not ret:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # -------- RED COLOR MASK --------
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.medianBlur(mask1 + mask2, 7)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detected_label = None
        detected_conf = 0.0

        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            try:
                roi_resized = cv2.resize(roi, self.img_size)
                x_input = img_to_array(roi_resized) / 255.0
                x_input = np.expand_dims(x_input, axis=0)

                pred = self.model.predict(x_input, verbose=0)
                class_idx = np.argmax(pred)
                label = self.classes[class_idx]
                conf = float(pred[0][class_idx])

                if conf > detected_conf:
                    detected_label = label
                    detected_conf = conf

                cv2.rectangle(
                    frame, (x, y), (x+w, y+h), (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    f"{label} ({conf:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            except Exception as e:
                print("Prediction error:", e)

        # -------- LOGGING --------
        if detected_label and detected_conf > 0.7:
            now = datetime.datetime.now()
            if now - self.last_log_time > self.log_interval:
                self.log_detection(now, detected_label, detected_conf)
                self.last_log_time = now

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def log_detection(self, time, label, conf):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                label,
                f"{conf:.4f}"
            ])
