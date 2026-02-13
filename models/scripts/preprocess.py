import os
import json
import random
import shutil
from sklearn.model_selection import train_test_split

# Paths
RAW_DIR = '../../data/raw/images'
PROCESSED_DIR = '../../data/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Categories based on your labels
CATEGORIES = [
    {"id": 1, "name": "crack"},
    {"id": 2, "name": "chip"},
    {"id": 3, "name": "missing_spring"},
    {"id": 4, "name": "spring"},
    {"id": 5, "name": "normal"}
]

# Folder mapping (update if folders renamed)
FOLDERS = ['crack', 'chip', 'missing_spring', 'spring', 'normal']  # Use your exact folder names

# Collect image-JSON pairs
pairs = []
for folder in FOLDERS:
    folder_path = os.path.join(RAW_DIR, folder)
    for json_file in [f for f in os.listdir(folder_path) if f.endswith('.json')]:
        json_path = os.path.join(folder_path, json_file)
        img_path = json_path.replace('.json', '.png')
        if os.path.exists(img_path):
            pairs.append((img_path, json_path))

print(f'Found {len(pairs)} pairs.')

# Function to create COCO JSON
def create_coco(pairs_subset, split):
    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    ann_id = 1
    for img_id, (img_path, json_path) in enumerate(pairs_subset):
        with open(json_path, 'r') as f:
            data = json.load(f)
        coco["images"].append({
            "id": img_id,
            "file_name": os.path.basename(img_path),
            "height": data["imageHeight"],
            "width": data["imageWidth"]
        })
        for shape in data.get("shapes", []):
            label = shape["label"]
            cat_id = next((c["id"] for c in CATEGORIES if c["name"] == label), None)
            if cat_id is None:
                continue
            points = shape["points"]
            segmentation = [coord for pt in points for coord in pt]
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "segmentation": [segmentation],
                "area": bbox[2] * bbox[3],  # Approx
                "bbox": bbox,
                "iscrowd": 0
            })
            ann_id += 1
    split_dir = os.path.join(PROCESSED_DIR, split)
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, 'instances.json'), 'w') as f:
        json.dump(coco, f)
    for img_path, _ in pairs_subset:
        shutil.copy(img_path, split_dir)
    print(f'{split} created with {len(pairs_subset)} images.')

# Split (80/10/10)
random.shuffle(pairs)
train_pairs, temp_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, random_state=42)

create_coco(train_pairs, 'train')
create_coco(val_pairs, 'val')
create_coco(test_pairs, 'test')