#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

YOLO_W = 640
YOLO_H = 640

class VisionNode:

    def __init__(self):
        rospy.init_node("vision_node")
        rospy.loginfo("Vision node (RGB) started")

        self.bridge = CvBridge()

        rospy.Subscriber(
            "/ur5/zed_node/left/image_rect_color",
            Image,
            self.image_callback,
            queue_size=1
        )

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(e)
            return

        h, w, _ = cv_image.shape
        rospy.loginfo_throttle(2.0, f"Original image size: {w}x{h}")

        resized = cv2.resize(cv_image, (YOLO_W, YOLO_H))
        rospy.loginfo_throttle(2.0, f"Resized to YOLO: {YOLO_W}x{YOLO_H}")

        # SOLO DEBUG: mostra immagine
        cv2.imshow("ZED RGB (YOLO input)", resized)
        cv2.waitKey(1)

if __name__ == "__main__":
    VisionNode()
    rospy.spin()