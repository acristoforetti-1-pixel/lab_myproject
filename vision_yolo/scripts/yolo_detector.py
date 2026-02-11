#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from ultralytics import YOLO
import yaml

class YoloDetector:
    def __init__(self):
        rospy.init_node("yolo_detector")
        self.bridge = CvBridge()

        # Load camera intrinsics
        intrinsics_path = rospy.get_param("~camera_intrinsics")
        with open(intrinsics_path, 'r') as f:
            intr = yaml.safe_load(f)
        K = intr["camera_matrix"]["data"]
        self.fx, self.fy = K[0], K[4]
        self.cx, self.cy = K[2], K[5]

        # YOLO model
        model_path = rospy.get_param("~model_path")
        self.model = YOLO(model_path)

        # Subscribers
        self.sub_img = rospy.Subscriber(
            "/zed/rgb/image_rect_color",
            Image,
            self.image_callback,
            queue_size=1
        )

        # Depth for 3D
        self.sub_depth = rospy.Subscriber(
            "/zed/depth/depth_registered",
            Image,
            self.depth_callback,
            queue_size=1
        )

        # Publisher for block position
        self.pub_pos = rospy.Publisher(
            "/blocks/position",
            PointStamped,
            queue_size=10
        )

        self.latest_depth = None

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def image_callback(self, msg):
        if self.latest_depth is None:
            return
        
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_resized = cv2.resize(frame, (640, 640))

        results = self.model(img_resized)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Convert back to original resolution
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 640
            x1 *= scale_x; x2 *= scale_x
            y1 *= scale_y; y2 *= scale_y
            
            cx = int((x1 + x2) / 2)
            cy = int((1*y1 + 1*y2) / 2)

            # Depth at center pixel
            Z = float(self.latest_depth[cy, cx])
            if Z == 0 or np.isnan(Z):
                continue

            # Pixel ? camera coordinates
            X = (cx - self.cx) * Z / self.fx
            Y = (cy - self.cy) * Z / self.fy

            pt = PointStamped()
            pt.header.stamp = rospy.Time.now()
            pt.header.frame_id = "zed_frame"
            pt.point.x = X
            pt.point.y = Y
            pt.point.z = Z

            self.pub_pos.publish(pt)

            # Show detections
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)

        cv2.imshow("YOLO detections", frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    YoloDetector()
    rospy.spin()

