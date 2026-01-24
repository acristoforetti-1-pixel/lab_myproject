#!/usr/bin/env python3
"""
check_yolo_dataset_recursive.py

Versione aggiornata dello script di verifica che:
- scansiona ricorsivamente images/<split>/...
- per ogni immagine trova labels/<split>/stesso/rel/path/file.txt
- disegna le bbox e salva output preservando la gerarchia relativa in output/
"""
import os
import random
import argparse
from PIL import Image, ImageDraw, ImageFont
import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

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

def collect_images_recursive(img_root):
    imgs = []
    for root, _, files in os.walk(img_root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, img_root)
                imgs.append((full, rel))
    return imgs

def main(args):
    random.seed(42)

    class_names = load_classes(args.data_yaml)

    img_root = os.path.join(args.dataset, "images", args.split)
    lbl_root = os.path.join(args.dataset, "labels", args.split)

    if not os.path.exists(img_root):
        print("Nessuna directory immagini trovata:", img_root)
        return

    imgs = collect_images_recursive(img_root)
    if not imgs:
        print("Nessuna immagine trovata in", img_root)
        return

    sample_n = min(args.num_samples, len(imgs))
    sampled = random.sample(imgs, sample_n)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for full_path, rel_path in sampled:
        lbl_rel = os.path.splitext(rel_path)[0] + ".txt"
        lbl_path = os.path.join(lbl_root, lbl_rel)

        img = Image.open(full_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        W, H = img.size

        if not os.path.exists(lbl_path):
            print(f"Label mancante per {rel_path}  -> {lbl_path}")
            out_img_path = os.path.join(args.output, rel_path)
            os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
            img.save(out_img_path)
            continue

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Formato label errato in {lbl_path}: {line.strip()}")
                continue
            cid, cx, cy, w, h = parts
            try:
                cid = int(cid)
                cx, cy, w, h = map(float, (cx, cy, w, h))
            except Exception:
                print(f"Valori label non numerici in {lbl_path}: {line.strip()}")
                continue
            if cid < 0 or cid >= len(class_names):
                print(f"Class ID fuori range ({cid}) in {lbl_path}")
                continue

            minx, miny, maxx, maxy = yolo_to_pixel(cx, cy, w, h, W, H)
            minx = max(0, min(minx, W - 1))
            miny = max(0, min(miny, H - 1))
            maxx = max(0, min(maxx, W - 1))
            maxy = max(0, min(maxy, H - 1))

            draw.rectangle([minx, miny, maxx, maxy], width=2)
            label = class_names[cid]
            draw.text((minx, max(miny - 15, 0)), label, font=font)

        out_img_path = os.path.join(args.output, rel_path)
        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        img.save(out_img_path)

    print("Debug completato")
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
