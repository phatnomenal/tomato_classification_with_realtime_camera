import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model("../models/mobilenet_tomato_finetune.h5")

# Input size model
img_size = (150, 150)

# Mapping class index → label
classes = ['fully-ripe', 'semi-ripe', 'unripe']  # kiểm tra train_data.class_indices nếu cần

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame để model predict
    img = cv2.resize(frame, img_size)
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Dự đoán lớp
    pred = model.predict(x)
    class_idx = np.argmax(pred)
    label = classes[class_idx]
    confidence = pred[0][class_idx]

    # Hiển thị label lên frame
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Tomato Ripeness Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
