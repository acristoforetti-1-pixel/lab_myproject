import os
import json
import shutil
from PIL import Image
import random

# ---------------- CONFIG ----------------
classes = [
    "X1-Y1-Z2",
    "X1-Y2-Z1",
    "X1-Y2-Z2",
    "X1-Y2-Z2-CHAMFER",
    "X1-Y2-Z2-TWINFILLET",
    "X1-Y3-Z2",
    "X1-Y3-Z2-FILLET",
    "X1-Y4-Z1",
    "X1-Y4-Z2",
    "X2-Y2-Z2",
    "X2-Y2-Z2-FILLET"
]

class_to_id = {c: i for i, c in enumerate(classes)}

OUT = "yolo_dataset"
IMG_DIR = os.path.join(OUT, "images")
LBL_DIR = os.path.join(OUT, "labels")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

samples = []

# ---------------- CONVERT ----------------
for assign in ["assign1", "assign2", "assign3"]:
    if not os.path.exists(assign):
        continue

    for scene in os.listdir(assign):
        scene_path = os.path.join(assign, scene)
        if not os.path.isdir(scene_path):
            continue

        for view in range(10):
            img_path = os.path.join(scene_path, f"view={view}_bbox.jpeg")
            json_path = os.path.join(scene_path, f"view={view}.json")

            if not os.path.exists(img_path) or not os.path.exists(json_path):
                continue

            with open(json_path) as f:
                data = json.load(f)

            img = Image.open(img_path)
            W, H = img.size

            labels = []

            for obj in data.values():
                if "bbox" not in obj or "y" not in obj:
                    continue

                cls = obj["y"]
                if cls not in class_to_id:
                    continue

                x1, y1, x2, y2 = obj["bbox"]

                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H

                if bw <= 0 or bh <= 0:
                    continue

                labels.append(
                    f"{class_to_id[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                )

            if not labels:
                continue

            name = f"{assign}_{scene}_view{view}"
            shutil.copy(img_path, os.path.join(IMG_DIR, name + ".jpg"))

            with open(os.path.join(LBL_DIR, name + ".txt"), "w") as f:
                f.write("\n".join(labels))

            samples.append(name)

# ---------------- SPLIT ----------------
random.shuffle(samples)
split = int(0.8 * len(samples))

for s in samples[:split]:
    os.makedirs(f"{OUT}/images/train", exist_ok=True)
    os.makedirs(f"{OUT}/labels/train", exist_ok=True)
    shutil.move(f"{IMG_DIR}/{s}.jpg", f"{OUT}/images/train/{s}.jpg")
    shutil.move(f"{LBL_DIR}/{s}.txt", f"{OUT}/labels/train/{s}.txt")

for s in samples[split:]:
    os.makedirs(f"{OUT}/images/val", exist_ok=True)
    os.makedirs(f"{OUT}/labels/val", exist_ok=True)
    shutil.move(f"{IMG_DIR}/{s}.jpg", f"{OUT}/images/val/{s}.jpg")
    shutil.move(f"{LBL_DIR}/{s}.txt", f"{OUT}/labels/val/{s}.txt")

print("✅ YOLO dataset ready")
