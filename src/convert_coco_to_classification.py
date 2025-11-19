import os
import json
from PIL import Image

# ==== CONFIG ====
datasets = ["train", "test"]
input_root = "../data/raw"
output_root = "../data/processed"

# category_id trong COCO -> tên lớp
label_map = {
    1: "unripe",
    2: "semi-ripe",
    3: "fully-ripe"
}

def ensure_dirs():
    for cls in label_map.values():
        os.makedirs(os.path.join(output_root, cls), exist_ok=True)

def process_dataset(split):
    print(f"Processing {split} ...")

    img_dir = os.path.join(input_root, split, "images")
    ann_file = os.path.join(input_root, split, "annotations", "instances.json")

    with open(ann_file, "r") as f:
        coco = json.load(f)

    # Map image_id -> file name
    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        label_id = ann["category_id"]
        bbox = ann["bbox"]  # [x, y, w, h]

        if label_id not in label_map:
            continue

        x, y, w, h = map(int, bbox)
        class_name = label_map[label_id]

        img_path = os.path.join(img_dir, id_to_filename[image_id])
        img = Image.open(img_path).convert("RGB")

        # Crop bounding box
        crop = img.crop((x, y, x + w, y + h))

        # Save cropped image
        save_name = f"{image_id}_{ann['id']}.jpg"
        save_path = os.path.join(output_root, class_name, save_name)
        crop.save(save_path)

    print(f"Done {split}.")

if __name__ == "__main__":
    ensure_dirs()
    for d in datasets:
        process_dataset(d)
