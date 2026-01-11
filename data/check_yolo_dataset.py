#!/usr/bin/env python3
"""
check_yolo_dataset.py

Script di controllo per dataset YOLO:
- seleziona N immagini random
- disegna bounding box YOLO
- salva immagini annotate in debug_check/
"""

import os
import random
import argparse
from PIL import Image, ImageDraw, ImageFont
import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png")

def load_classes(data_yaml):
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)
    return data["names"]

def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    minx = (cx * img_w) - bw / 2
    miny = (cy * img_h) - bh / 2
    maxx = minx + bw
    maxy = miny + bh
    return minx, miny, maxx, maxy

def main(args):
    random.seed(42)

    class_names = load_classes(args.data_yaml)

    img_dir = os.path.join(args.dataset, "images", args.split)
    lbl_dir = os.path.join(args.dataset, "labels", args.split)

    images = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)]
    if not images:
        print("❌ Nessuna immagine trovata in", img_dir)
        return

    sample_imgs = random.sample(images, min(args.num_samples, len(images)))

    os.makedirs(args.output, exist_ok=True)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for img_name in sample_imgs:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        W, H = img.size

        if not os.path.exists(lbl_path):
            print(f"⚠️ Label mancante per {img_name}")
            img.save(os.path.join(args.output, img_name))
            continue

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"⚠️ Formato label errato in {lbl_path}: {line}")
                continue

            cid, cx, cy, w, h = parts
            cid = int(cid)
            cx, cy, w, h = map(float, (cx, cy, w, h))

            if cid >= len(class_names):
                print(f"⚠️ Class ID fuori range ({cid}) in {lbl_path}")
                continue

            minx, miny, maxx, maxy = yolo_to_pixel(cx, cy, w, h, W, H)

            # clamp
            minx = max(0, min(minx, W - 1))
            miny = max(0, min(miny, H - 1))
            maxx = max(0, min(maxx, W - 1))
            maxy = max(0, min(maxy, H - 1))

            draw.rectangle([minx, miny, maxx, maxy], width=2)
            label = class_names[cid]
            draw.text((minx, max(miny - 15, 0)), label, font=font)

        out_path = os.path.join(args.output, img_name)
        img.save(out_path)

    print("✅ Debug completato")
    print(f"Immagini annotate salvate in: {args.output}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_yolo", help="Root dataset YOLO")
    ap.add_argument("--data-yaml", default="dataset_yolo/data.yaml", help="Path data.yaml")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--num-samples", type=int, default=20, help="Numero immagini da controllare")
    ap.add_argument("--output", default="debug_check", help="Cartella output immagini annotate")
    args = ap.parse_args()
    main(args)
