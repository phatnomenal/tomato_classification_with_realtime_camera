# 🍅 Tomato Ripeness Detection

A full pipeline for detecting and classifying tomato ripeness using:

* **Faster R-CNN** → tomato object detection
* **MobileNetV2 (fine-tuned)** → ripeness classification
* **Real-time webcam demo** → bounding boxes + labels + confidence

Ripeness levels:

* **unripe** (green)
* **semi-ripe** (orange)
* **fully-ripe** (red)

This project supports training, preprocessing, conversion from COCO annotations, and real-time inference.

---

## 📁 Project Structure

```
project_root/
│
├─ data/
│   ├─ raw/                    # Original images + COCO annotations
│   └─ processed/              # Cropped images for 3-class classification
│
├─ models/
│   ├─ mobilenet_tomato_classifier.h5     # Base MobileNetV2
│   └─ mobilenet_tomato_finetune.h5       # Fine-tuned MobileNetV2
│
├─ notebooks/               
│   ├─ MobileNet_V2_training.ipynb           # MobileNetV2 base training 
│
├─ src/
│    ├─ check.py                           # Check the labels
│   └─ convert_coco_to_classification.py  # Convert COCO → cropped dataset
│
├─ demos/
│   ├─ camera_demo.py                     # MobileNetV2 webcam classifier
│   └─ camera_demo_fasterrcnn.py          # Faster R-CNN + MobileNetV2 demo
├─ dáhboard/
│   ├─ app.py             # Flask application
│   └─ camera.py          # setup camera
│
└─ requirements.txt
```

---

## ⚡ Environment Requirements

* Python **3.8+**
* TensorFlow **2.x**
* PyTorch + **torchvision**
* OpenCV
* Pillow

### Install:

```bash
pip install tensorflow torch torchvision opencv-python pillow
```

---

## 📝 Usage Guide

### **1️⃣ Train MobileNetV2 Classifier (optional)**

Open the fine-tuning notebook:

```bash
jupyter notebook notebooks/MobileNet_V2_training.ipynb
```

The 2 trained model will be saved to the directory and we will use:

```
models/mobilenet_tomato_finetune.h5
```

---

### **2️⃣ Run Webcam Demo – MobileNetV2 Only**

```bash
python demos/camera_demo.py
```

This runs:

* Webcam livestream
* Full-frame classification
* Displays predicted ripeness + confidence

---

### **3️⃣ Run Webcam Demo – Faster R-CNN + MobileNetV2**

```bash
python demos/camera_demo_fasterrcnn.py
```

Pipeline:

1. Faster R-CNN detects tomato bounding boxes
2. Each crop is classified with MobileNetV2
3. Bounding boxes + label + confidence displayed in real-time
4. Press **q** to quit

---

## 💡 Performance & Optimization Tips

* Use **good lighting** → improves detection accuracy
* If FPS is low → **resize frame** before sending to Faster R-CNN
* For highest accuracy → fine-tune Faster R-CNN on your tomato dataset
* Convert model to **TensorFlow Lite** for IoT devices (optional)

---

## 📚 Resources

* Tomato dataset: tomatOD
* MobileNetV2 pretrained on ImageNet
* Faster R-CNN pretrained on COCO

---

## 👤 Author

AIoT project for real-time tomato ripeness classification & detection.
