import cv2

img = cv2.imread("assign1/scene1/view=0_bbox.jpeg")

if img is None:
    print("Image not found")
    exit()

cv2.imshow("BBOX IMAGE", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
