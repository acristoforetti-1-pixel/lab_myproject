import os
import json
import cv2
import random

# ---------------- CONFIG ----------------
SRC_ROOT = "assign1"
OUT_ROOT = "yolo_dataset"
TRAIN_SPLIT = 0.8

IMG_OUT_TRAIN = os.path.join(OUT_ROOT, "images/train")
IMG_OUT_VAL   = os.path.join(OUT_ROOT, "images/val")
LBL_OUT_TRAIN = os.path.join(OUT_ROOT, "labels/train")
LBL_OUT_VAL   = os.path.join(OUT_ROOT, "labels/val")

os.makedirs(IMG_OUT_TRAIN, exist_ok=True)
os.makedirs(IMG_OUT_VAL, exist_ok=True)
os.makedirs(LBL_OUT_TRAIN, exist_ok=True)
os.makedirs(LBL_OUT_VAL, exist_ok=True)

# ----------------------------------------

def convert_bbox(bbox, w, h):
    xmin, ymin, xmax, ymax = bbox
    x = ((xmin + xmax) / 2) / w
    y = ((ymin + ymax) / 2) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return x, y, bw, bh

# raccogli classi
classes = set()
samples = []

for scene in os.listdir(SRC_ROOT):
    scene_path = os.path.join(SRC_ROOT, scene)
    if not os.path.isdir(scene_path):
        continue

    for f in os.listdir(scene_path):
        if f.endswith(".json"):
            samples.append((scene_path, f))
            with open(os.path.join(scene_path, f)) as jf:
                data = json.load(jf)
                for obj in data.values():
                    classes.add(obj["y"])

classes = sorted(list(classes))
class_to_id = {c: i for i, c in enumerate(classes)}

print("Classes:", class_to_id)

random.shuffle(samples)
split = int(len(samples) * TRAIN_SPLIT)
train_samples = samples[:split]
val_samples = samples[split:]

def process(samples, img_out, lbl_out):
    for scene_path, json_file in samples:
        idx = json_file.replace(".json", "")
        img_file = idx + ".jpeg"

        img_path = os.path.join(scene_path, img_file)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        label_lines = []

        with open(os.path.join(scene_path, json_file)) as jf:
            data = json.load(jf)

        for obj in data.values():
            cls = obj["y"]
            bbox = obj["bbox"]

            if bbox[1] > bbox[3]:
                bbox[1], bbox[3] = bbox[3], bbox[1]

            cid = class_to_id[cls]
            x, y, bw, bh = convert_bbox(bbox, w, h)

            label_lines.append(f"{cid} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

        out_name = f"{scene_path.split('/')[-1]}_{idx}"
        cv2.imwrite(os.path.join(img_out, out_name + ".jpg"), img)

        with open(os.path.join(lbl_out, out_name + ".txt"), "w") as lf:
            lf.write("\n".join(label_lines))

process(train_samples, IMG_OUT_TRAIN, LBL_OUT_TRAIN)
process(val_samples, IMG_OUT_VAL, LBL_OUT_VAL)

# dataset.yaml
with open(os.path.join(OUT_ROOT, "dataset.yaml"), "w") as f:
    f.write(f"""
path: {os.path.abspath(OUT_ROOT)}
train: images/train
val: images/val

names:
""")
    for c in classes:
        f.write(f"  - {c}\n")

print("YOLO dataset created.")