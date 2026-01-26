#!/usr/bin/env python3
"""perception_6d_node.py (ULTRALYTICS-only, multi-detection + post-filter debug image)

Behavior changes implemented now:
 - Process all detections returned by the detector (not only the highest-confidence one).
 - For each detection: sample the pointcloud at the bbox center, transform to base frame, apply the single Z-axis filter (z >= -1.0 kept), and publish poses only for accepted detections.
 - The debug image `/perception/debug/image_raw` now shows ALL detected bboxes: green boxes are ACCEPTED (published in /vision/object_pose), red boxes are REJECTED (filtered out). This prevents the arm from "starving" the system visually while allowing you to see why detections were rejected.

No new ROS params were added; z-threshold remains hardcoded as before (z >= -1.0 accepted).

This file replaces the previous canvas contents.
"""

from __future__ import annotations
import rospy
import threading
import math
import numpy as np

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import PoseStamped, PointStamped
import sensor_msgs.point_cloud2 as pc2

from cv_bridge import CvBridge, CvBridgeError

import tf2_ros
from tf2_geometry_msgs import do_transform_point
import tf.transformations as tft
from visualization_msgs.msg import Marker

import cv2

# ultralytics loader only
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ----------------- helpers
def euler_to_quat(roll, pitch, yaw):
    q = tft.quaternion_from_euler(roll, pitch, yaw)
    return q

def is_finite_point(pt):
    return (pt is not None and not any([math.isinf(v) or math.isnan(v) for v in pt]))

class Perception6DNode:
    def __init__(self):
        rospy.init_node("perception_6d_node", anonymous=False)

        # params (kept minimal)
        self.model_path = rospy.get_param("~model_path", "/root/ros_ws/src/lab_myproject/data/runs/detect/train/weights/best.pt")
        self.image_topic = rospy.get_param("~image_topic", "/ur5/zed_node/right_raw/image_raw_color")
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "/ur5/zed_node/point_cloud/cloud_registered")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/ur5/zed_node/right_raw/camera_info")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        # detection tuning
        self.conf_thresh = rospy.get_param("~conf_thresh", 0.45)
        self.imgsz = rospy.get_param("~imgsz", 640)

        # visualization
        self.marker_size = rospy.get_param("~marker_size", 0.06)

        # z filter threshold (hardcoded as requested)
        self.z_threshold = -1.0  # detections with z < -1.0 (in base_frame) are discarded

        # publishers
        self.pub_pose = rospy.Publisher("/vision/object_pose", PoseStamped, queue_size=5)
        self.pub_rpy = rospy.Publisher("/vision/object_rpy", Float64MultiArray, queue_size=5)
        self.pub_name = rospy.Publisher("/vision/object_name", String, queue_size=5)
        self.pub_marker = rospy.Publisher("/vision/detection_marker", Marker, queue_size=5)
        self.pub_debug_img = rospy.Publisher("/perception/debug/image_raw", Image, queue_size=1)

        # state
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_pc = None
        self.latest_img = None
        self.latest_caminfo = None

        # tf
        self.tf_buf = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        # ultralytics required
        if YOLO is None:
            rospy.logerr("ultralytics not found. Please pip install ultralytics and restart.")
            rospy.signal_shutdown("ultralytics_missing")
            return

        # load model
        try:
            self.model = YOLO(self.model_path)
            rospy.loginfo("Loaded model using ultralytics.YOLO: %s", self.model_path)
        except Exception as e:
            rospy.logerr("Failed to load model with ultralytics.YOLO: %s", e)
            rospy.signal_shutdown("model_load_failed")
            return

        # subscribers
        rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.pc_cb, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.caminfo_cb, queue_size=1)

        rospy.loginfo("Perception6DNode ready. publishing poses -> %s", "/vision/object_pose")

    # ---- callbacks
    def pc_cb(self, msg: PointCloud2):
        with self.lock:
            self.latest_pc = msg

    def caminfo_cb(self, msg: CameraInfo):
        with self.lock:
            self.latest_caminfo = msg

    def image_cb(self, msg: Image):
        # snapshot latest pc
        with self.lock:
            self.latest_img = msg
            pc = self.latest_pc

        # convert image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn("cv_bridge error: %s", e)
            return

        # run model
        try:
            res = self.model(cv_img, imgsz=self.imgsz)
        except Exception as e:
            rospy.logwarn("Model inference failed: %s", e)
            return

        # parse ultralytics result
        try:
            r = res[0]
            boxes = getattr(r, 'boxes', None)
            names = getattr(r, 'names', {})
            if boxes is None or len(boxes) == 0:
                return

            # prepare debug image (draw ALL boxes, color later according to accept/reject)
            img_dbg = cv_img.copy()

            # process all boxes (not only best)
            accepted_any = False
            for b in boxes:
                try:
                    conf = float(b.conf[0]) if hasattr(b, 'conf') else float(b.conf)
                except Exception:
                    conf = 0.0
                # skip low confidence boxes
                if conf < self.conf_thresh:
                    continue

                try:
                    xyxy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy, 'cpu') else np.array(b.xyxy[0])
                    cls = int(b.cls[0]) if hasattr(b, 'cls') else int(b.cls)
                except Exception as e:
                    rospy.logwarn("Failed to parse box coords: %s", e)
                    continue

                name = names.get(cls, str(cls)) if isinstance(names, dict) else (names[cls] if names is not None and cls < len(names) else str(cls))
                x1, y1, x2, y2 = xyxy.tolist()
                xcenter = int((x1 + x2) / 2.0)
                ycenter = int((y1 + y2) / 2.0)

                # sample single center point from pointcloud (camera frame)
                pt_cam = None
                if pc is not None:
                    pt_cam = self.sample_point_from_pc(pc, xcenter, ycenter, 3)

                accepted = False
                if is_finite_point(pt_cam):
                    # build PointStamped in camera frame
                    ps = PointStamped()
                    ps.header = pc.header if pc is not None else msg.header
                    ps.point.x, ps.point.y, ps.point.z = pt_cam

                    # transform to base_frame via lookup+do_transform_point
                    try:
                        t = self.tf_buf.lookup_transform(self.base_frame, ps.header.frame_id, rospy.Time(0), rospy.Duration(0.5))
                        ps_base = do_transform_point(ps, t)

                        z_base = ps_base.point.z
                        # SIMPLE z-axis filter: discard detections with z < -1.0
                        if z_base >= self.z_threshold:
                            accepted = True
                        else:
                            accepted = False
                    except Exception as e:
                        rospy.logwarn("TF transform failed (lookup+do_transform) for box: %s", e)
                        accepted = False
                else:
                    rospy.logdebug("No valid pointcloud point for bbox center (u=%d v=%d)", xcenter, ycenter)
                    accepted = False

                # draw bbox: green if accepted, red if rejected
                col = (0,255,0) if accepted else (0,0,255)
                cv2.rectangle(img_dbg, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                cv2.circle(img_dbg, (int(xcenter), int(ycenter)), 3, (255,255,0), -1)
                label = f"{name} {conf:.2f}"
                cv2.putText(img_dbg, label, (max(5,int(x1)), max(20,int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

                # if accepted, publish pose+name
                if accepted:
                    pose = PoseStamped()
                    pose.header.stamp = rospy.Time.now()
                    pose.header.frame_id = self.base_frame
                    pose.pose.position.x = ps_base.point.x
                    pose.pose.position.y = ps_base.point.y
                    pose.pose.position.z = ps_base.point.z
                    roll = math.pi
                    pitch = 0.0
                    yaw = 0.0
                    q = euler_to_quat(roll, pitch, yaw)
                    pose.pose.orientation.x = q[0]
                    pose.pose.orientation.y = q[1]
                    pose.pose.orientation.z = q[2]
                    pose.pose.orientation.w = q[3]

                    nm = String(); nm.data = name; self.pub_name.publish(nm)
                    m = Float64MultiArray(); m.data = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, roll, pitch, yaw]; self.pub_rpy.publish(m)
                    self.pub_pose.publish(pose)
                    self.publish_marker(pose, name, conf)
                    accepted_any = True

            # publish debug image after processing all boxes
            try:
                img_msg = self.bridge.cv2_to_imgmsg(img_dbg, encoding="bgr8")
                img_msg.header = msg.header
                self.pub_debug_img.publish(img_msg)
            except Exception as e:
                rospy.logwarn_throttle(10, "Failed to publish debug image: %s", e)

            if not accepted_any:
                rospy.logdebug("No detections passed the z-filter on this frame.")

        except Exception as e:
            rospy.logwarn("Parsing detection result failed: %s", e)
            return

    # ---- helpers
    def sample_point_from_pc(self, pc_msg, u, v, radius):
        # read single pixel (with neighbor search)
        if pc_msg is None:
            return None
        width = pc_msg.width
        height = pc_msg.height
        if width == 0 or height == 0:
            return None
        u = max(0, min(width - 1, int(u)))
        v = max(0, min(height - 1, int(v)))
        attempts = [(0,0)]
        for r in range(1, radius+1):
            for dx in range(-r, r+1):
                dy = r
                attempts.append((dx, dy))
                attempts.append((dx, -dy))
            for dy in range(-r+1, r):
                dx = r
                attempts.append((dx, dy))
                attempts.append((-dx, dy))
        for dx, dy in attempts:
            uu = u + dx
            vv = v + dy
            if uu < 0 or uu >= width or vv < 0 or vv >= height:
                continue
            try:
                for p in pc2.read_points(pc_msg, field_names=('x','y','z'), skip_nans=False, uvs=[(uu,vv)]):
                    pt = (float(p[0]), float(p[1]), float(p[2]))
                    if is_finite_point(pt):
                        return pt
            except Exception:
                return None
        return None

    def publish_marker(self, pose: PoseStamped, name: str, conf: float):
        m = Marker()
        m.header.frame_id = pose.header.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = "vision_det"
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose = pose.pose
        m.scale.x = self.marker_size
        m.scale.y = self.marker_size
        m.scale.z = self.marker_size
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.8
        m.lifetime = rospy.Duration(0.5)
        self.pub_marker.publish(m)

        mt = Marker()
        mt.header.frame_id = pose.header.frame_id
        mt.header.stamp = rospy.Time.now()
        mt.ns = "vision_det_text"
        mt.id = 1
        mt.type = Marker.TEXT_VIEW_FACING
        mt.action = Marker.ADD
        mt.pose = pose.pose
        mt.pose.position.z += (self.marker_size * 1.2)
        mt.scale.z = max(0.04, self.marker_size * 0.8)
        mt.color.r = 1.0
        mt.color.g = 1.0
        mt.color.b = 1.0
        mt.color.a = 0.9
        mt.text = f"{name} {conf:.2f}"
        mt.lifetime = rospy.Duration(0.5)
        self.pub_marker.publish(mt)

# ---------- main
def main():
    Perception6DNode()
    rospy.spin()

if __name__ == "__main__":
    main()
