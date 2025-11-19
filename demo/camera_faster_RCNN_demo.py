import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load MobileNetV2 classifier
classifier = load_model("../models/mobilenet_tomato_finetune.h5")
classes = ['fully-ripe', 'semi-ripe', 'unripe']  # mapping từ train_data

# Load Faster R-CNN (pretrained COCO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = fasterrcnn_resnet50_fpn(pretrained=True)
detector.eval().to(device)

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    img_tensor = F.to_tensor(frame).to(device)

    # Detect objects
    with torch.no_grad():
        outputs = detector([img_tensor])

    # Lấy boxes và scores
    boxes = outputs[0]['boxes']
    scores = outputs[0]['scores']
    labels = outputs[0]['labels']

    for box, score, label in zip(boxes, scores, labels):
        if score < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box)
        crop = orig[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Resize và predict ripeness
        crop_resized = cv2.resize(crop, (150,150))
        x = img_to_array(crop_resized)/255.0
        x = np.expand_dims(x, axis=0)
        pred = classifier.predict(x)
        class_idx = np.argmax(pred)
        label_text = f"{classes[class_idx]} ({pred[0][class_idx]:.2f})"

        # Draw bounding box + label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label_text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Tomato Ripeness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
