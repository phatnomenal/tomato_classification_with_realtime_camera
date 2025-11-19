🍅 Tomato Ripeness Detection

Mô tả dự án:
Dự án sử dụng Faster R-CNN để phát hiện các quả cà chua trong hình ảnh/video, kết hợp với MobileNetV2 fine-tune để phân loại độ chín của từng quả:

unripe – chưa chín

semi-ripe – chín một phần

fully-ripe – chín hoàn toàn

Hỗ trợ realtime webcam demo để quan sát kết quả trực tiếp.

📁 Cấu trúc thư mục

project_root/
│
├─ data/
│   ├─ raw/                 # ảnh gốc + annotation COCO
│   └─ processed/           # ảnh crop cho classifier 3 lớp
│
├─ models/
│   ├─ mobilenet_tomato_classifier.h5        # model MobileNetV2 ban đầu
│   └─ mobilenet_tomato_finetune.h5         # model fine-tune
│
├─ notebooks/
│   ├─ 01-training.ipynb
│   ├─ 02-train_mobilenet.ipynb
│   └─ 03-train_mobilenet_finetune.ipynb
│
├─ src/
│   └─ convert_coco_to_classification.py
│
├─ demos/
│   ├─ camera_demo.py                         # MobileNetV2 webcam demo
│   └─ camera_demo_fasterrcnn.py             # Faster R-CNN + MobileNetV2
│
└─ requirements.txt


⚡ Yêu cầu môi trường

Python 3.8+

TensorFlow 2.x

PyTorch + torchvision

OpenCV

Pillow

Cài đặt các thư viện:

pip install tensorflow torch torchvision opencv-python pillow

📝 Hướng dẫn sử dụng
1️⃣ Train model MobileNetV2 (tuỳ chọn)

Chạy notebook fine-tune:

jupyter notebook notebooks/03-train_mobilenet_finetune.ipynb


Model sẽ được lưu tại:

models/mobilenet_tomato_finetune.h5

2️⃣ Chạy webcam demo với MobileNetV2
python demos/camera_demo.py


Mở webcam, hiển thị label + confidence cho toàn frame

3️⃣ Chạy webcam demo với Faster R-CNN + MobileNetV2
python demos/camera_demo_fasterrcnn.py


Phát hiện từng quả cà chua (bounding box)

Crop → predict độ chín bằng MobileNetV2

Hiển thị trực tiếp bounding box + label + confidence

Nhấn q để thoát

💡 Gợi ý tối ưu

Giữ ánh sáng tốt → giúp Faster R-CNN detect chính xác

Nếu FPS quá thấp, resize frame trước khi detect

Có thể fine-tune Faster R-CNN với dataset cà chua để tăng độ chính xác

📚 Tài nguyên

Dataset cà chua: [Kaggle / GitHub]

MobileNetV2 pre-trained trên ImageNet

Faster R-CNN pre-trained COCO

📝 License

MIT License (hoặc tuỳ bạn đặt)
