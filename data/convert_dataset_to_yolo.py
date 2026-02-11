#!/usr/bin/env python3
/**
 * @file convert_dataset_to_yolo.py
 * @brief Converts a custom annotated dataset into YOLO format.
 *
 * This script scans a dataset containing JSON annotations (view=*.json)
 * and corresponding images, extracts object bounding boxes, converts them
 * into normalized YOLO format, and generates:
 *
 * - images/train and images/val directories
 * - labels/train and labels/val directories
 * - a data.yaml configuration file
 *
 * The relative directory structure (e.g., assignX/sceneY) is preserved
 * in order to avoid filename collisions.
 *
 * Supports optional scene-based splitting to ensure that all views
 * from the same scene belong to the same dataset split.
 */
import os
import json
import random
import shutil
import argparse
from collections import defaultdict
from PIL import Image
/**
 * @brief List of supported object class names.
 *
 * The index of each class in this list corresponds to its YOLO class ID.
 */
CLASS_NAMES = [
    "X1-Y1-Z2","X1-Y2-Z1","X1-Y2-Z2","X1-Y2-Z2-CHAMFER","X1-Y2-Z2-TWINFILLET",
    "X1-Y3-Z2","X1-Y3-Z2-FILLET","X1-Y4-Z1","X1-Y4-Z2","X2-Y2-Z2","X2-Y2-Z2-FILLET",
]
/**
 * @brief Mapping from class name to numeric YOLO class ID.
 */
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
/**
 * @brief Possible JSON keys that may contain class information.
 *
 * The script searches these keys when extracting object labels
 * from annotation files.
 */
POSSIBLE_CLASS_KEYS = ['label', 'class', 'y', 'name', 'type']
/**
 * @brief Computes an axis-aligned bounding box from polygon vertices.
 *
 * @param vertices List of (x, y) points.
 * @return Tuple (min_x, min_y, max_x, max_y).
 */
def bbox_from_vertices(vertices):
    xs = [float(v[0]) for v in vertices]
    ys = [float(v[1]) for v in vertices]
    return min(xs), min(ys), max(xs), max(ys)
/**
 * @brief Converts a pixel-space bounding box to YOLO normalized format.
 *
 * Clamps coordinates inside image boundaries and returns:
 * (center_x, center_y, width, height) normalized in [0,1].
 *
 * @param bbox Tuple (min_x, min_y, max_x, max_y) in pixels.
 * @param w Image width in pixels.
 * @param h Image height in pixels.
 * @return Normalized YOLO bbox or None if invalid.
 */
def normalize_bbox(bbox, w, h):
    minx, miny, maxx, maxy = bbox
    minx = max(0.0, min(minx, w - 1.0))
    maxx = max(0.0, min(maxx, w - 1.0))
    miny = max(0.0, min(miny, h - 1.0))
    maxy = max(0.0, min(maxy, h - 1.0))
    bw = maxx - minx
    bh = maxy - miny
    if bw <= 0 or bh <= 0:
        return None
    cx = (minx + bw / 2.0) / w
    cy = (miny + bh / 2.0) / h
    return cx, cy, bw / w, bh / h
/**
 * @brief Extracts the class name from a JSON object entry.
 *
 * Searches predefined keys and checks for valid class names.
 *
 * @param obj Dictionary representing one annotated object.
 * @return Class name string or None if not found.
 */
def extract_class_from_obj(obj):
    for k in POSSIBLE_CLASS_KEYS:
        if k in obj and isinstance(obj[k], str):
            return obj[k]
    for k,v in obj.items():
        if isinstance(v, str) and v in CLASS_TO_ID:
            return v
    return None
/**
 * @brief Processes a single JSON annotation file and generates a YOLO label file.
 *
 * For each valid object in the JSON file:
 * - Extracts class name
 * - Computes bounding box
 * - Normalizes to YOLO format
 * - Writes results into a .txt label file
 *
 * @param json_path Path to the input JSON annotation.
 * @param img_path Path to the associated image.
 * @param label_out Output path for the YOLO label file.
 * @return True if at least one valid label was generated, False otherwise.
 */
def process_view(json_path, img_path, label_out):
    try:
        img = Image.open(img_path)
        w, h = img.size
    except Exception as e:
        print(f"[WARN] Impossibile aprire immagine {img_path}: {e}")
        return False
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Impossibile leggere json {json_path}: {e}")
        return False

    lines = []
    for obj_key, obj in data.items():
        if not isinstance(obj, dict):
            continue
        class_name = extract_class_from_obj(obj)
        if class_name is None or class_name not in CLASS_TO_ID:
            continue
        verts = obj.get('vertices') or obj.get('3d_bbox_pixel_space') or obj.get('polygon') or None
        if not verts:
            bbox_field = obj.get('bbox')
            if bbox_field and len(bbox_field) >= 4:
                bx = [float(x) for x in bbox_field[:4]]
                minx = min(bx[0], bx[2]); maxx = max(bx[0], bx[2])
                miny = min(bx[1], bx[3]); maxy = max(bx[1], bx[3])
                bbox = (minx, miny, maxx, maxy)
                norm = normalize_bbox(bbox, w, h)
                if norm:
                    cid = CLASS_TO_ID[class_name]
                    lines.append(f"{cid} {norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f} {norm[3]:.6f}")
            continue
        try:
            if len(verts) == 2 and isinstance(verts[0], (list, tuple)):
                minx, miny = float(verts[0][0]), float(verts[0][1])
                maxx, maxy = float(verts[1][0]), float(verts[1][1])
                bbox = (minx, miny, maxx, maxy)
            else:
                bbox = bbox_from_vertices(verts)
            norm = normalize_bbox(bbox, w, h)
            if norm:
                cid = CLASS_TO_ID[class_name]
                lines.append(f"{cid} {norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f} {norm[3]:.6f}")
        except Exception as e:
            print(f"[WARN] errore nel calcolo bbox per {json_path} obj {obj_key}: {e}")
            continue

    if lines:
        os.makedirs(os.path.dirname(label_out), exist_ok=True)
        with open(label_out, 'w') as f:
            f.write("\n".join(lines) + "\n")
        return True
    return False
/**
 * @brief Collects dataset samples from the input directory.
 *
 * Searches recursively for view=*.json files and their corresponding images.
 *
 * If split_by_scene is enabled, samples are grouped by scene to ensure
 * consistent train/validation splitting.
 *
 * @param input_root Root directory of the dataset.
 * @param split_by_scene Whether to group samples by scene.
 * @return (groups, grouped_flag)
 */
def collect_samples(input_root, split_by_scene=False):
    samples = []
    for root, _, files in os.walk(input_root):
        json_files = [f for f in files if f.startswith('view=') and f.lower().endswith('.json')]
        if not json_files:
            continue
        rel = os.path.relpath(root, input_root)
        scene_id = rel
        for part in rel.split(os.sep)[::-1]:
            if part.lower().startswith('scene'):
                scene_id = part
                break
        for jf in json_files:
            json_path = os.path.join(root, jf)
            base = os.path.splitext(jf)[0]
            img_path = None
            for ext in ['.jpeg', '.jpg', '.png', '.bmp']:
                cand = os.path.join(root, base + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break
            if img_path:
                # store also the relative directory so we can reproduce hierarchy
                rel_dir = os.path.relpath(root, input_root)
                samples.append((json_path, img_path, scene_id, rel_dir))
    if split_by_scene:
        by_scene = defaultdict(list)
        for s in samples:
            by_scene[s[2]].append((s[0], s[1], s[3]))
        groups = list(by_scene.values())
        return groups, True
    else:
        # return flat list of tuples (json, img, rel_dir)
        return samples, False
/**
 * @brief Exports dataset samples into YOLO directory structure.
 *
 * Creates train/val splits, preserves relative directories,
 * generates image and label files, and writes data.yaml.
 *
 * @param groups List of samples or grouped samples.
 * @param grouped Indicates whether grouping by scene is enabled.
 * @param args Parsed command-line arguments.
 */
def export_samples_grouped(groups, grouped, args):
    img_out_root = os.path.join(args.output, 'images')
    lbl_out_root = os.path.join(args.output, 'labels')

    cnt_labels = 0

    if grouped:
        all_groups = groups
        random.shuffle(all_groups)
        split_idx = int(len(all_groups) * args.train_ratio)
        train_groups = all_groups[:split_idx]
        val_groups = all_groups[split_idx:]
        # flatten keeping rel_dir
        train_items = [item for g in train_groups for item in g]  # item = (json, img, rel_dir)
        val_items = [item for g in val_groups for item in g]
    else:
        # groups is actually the flat samples list: (json, img, rel_dir)
        all_items = groups
        random.shuffle(all_items)
        split_idx = int(len(all_items) * args.train_ratio)
        train_items = all_items[:split_idx]
        val_items = all_items[split_idx:]

    def handle_pair(json_path, img_path, rel_dir, split_name):
        nonlocal cnt_labels
        # preserve relative dir to avoid name collisions
        out_img_dir = os.path.join(img_out_root, split_name, rel_dir)
        out_lbl_dir = os.path.join(lbl_out_root, split_name, rel_dir)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)
        img_dst = os.path.join(out_img_dir, os.path.basename(img_path))
        lbl_dst = os.path.join(out_lbl_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        if args.copy_images:
            shutil.copy2(img_path, img_dst)
        else:
            try:
                if not os.path.exists(img_dst):
                    os.link(img_path, img_dst)
            except Exception:
                shutil.copy2(img_path, img_dst)
        ok = process_view(json_path, img_dst, lbl_dst)
        if ok:
            cnt_labels += 1

    for item in train_items:
        if grouped:
            j,i,rel = item
        else:
            j,i,rel = item
        handle_pair(j, i, rel, 'train')
    for item in val_items:
        if grouped:
            j,i,rel = item
        else:
            j,i,rel = item
        handle_pair(j, i, rel, 'val')

    yaml_path = os.path.join(args.output, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names:\n")
        for n in CLASS_NAMES:
            f.write(f"  - {n}\n")

    total_train = sum([len(files) for _,_,files in os.walk(os.path.join(img_out_root, 'train'))])
    total_val = sum([len(files) for _,_,files in os.walk(os.path.join(img_out_root, 'val'))])
    print(f"Totale immagini: train={total_train} val={total_val}")
    print(f"Label generate: {cnt_labels}")
    print("Output:", os.path.abspath(args.output))
    print("data.yaml ->", yaml_path)
/**
 * @brief Entry point of the dataset conversion script.
 *
 * Parses command-line arguments, collects dataset samples,
 * and triggers export to YOLO format.
 */
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='.', help='Cartella radice dei dati (default current dir)')
    p.add_argument('--output', '-o', default='dataset_yolo', help='Cartella di output')
    p.add_argument('--train-ratio', type=float, default=0.8, help='Percentuale train (default 0.8)')
    p.add_argument('--copy-images', action='store_true', help='Copia tutte le immagini in output/images/...')
    p.add_argument('--split-by-scene', action='store_true', help='Metti tutte le view della stessa scena nello stesso split')
    args = p.parse_args()

    random.seed(42)
    input_root = os.path.abspath(args.input)
    print("Input:", input_root)
    print("Output:", os.path.abspath(args.output))
    groups, grouped = collect_samples(input_root, split_by_scene=args.split_by_scene)
    if not groups:
        print("[ERRORE] Nessun file view=*.json trovato sotto", input_root)
        return
    export_samples_grouped(groups, grouped, args)

if __name__ == '__main__':
    main()
