import cv2
import os
import random

IMG_DIR = "yolo_dataset/images/train"
LBL_DIR = "yolo_dataset/labels/train"

IMG_EXT = [".jpg", ".jpeg", ".png"]

def load_random_sample():
    imgs = [f for f in os.listdir(IMG_DIR) if any(f.endswith(e) for e in IMG_EXT)]
    img_name = random.choice(imgs)
    lbl_name = img_name.replace(".jpg", ".txt").replace(".jpeg", ".txt")

    img = cv2.imread(os.path.join(IMG_DIR, img_name))
    h, w, _ = img.shape

    with open(os.path.join(LBL_DIR, lbl_name)) as f:
        lines = f.readlines()

    for line in lines:
        cls, cx, cy, bw, bh = map(float, line.split())

        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, str(int(cls)), (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("YOLO label check", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    load_random_sample()
