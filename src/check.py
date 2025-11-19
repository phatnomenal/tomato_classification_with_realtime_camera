import json

ann_file = "../data/raw/train/annotations/instances.json"

with open(ann_file) as f:
    coco = json.load(f)

category_ids = [ann["category_id"] for ann in coco["annotations"]]
print("Category IDs có trong dataset:", set(category_ids))
