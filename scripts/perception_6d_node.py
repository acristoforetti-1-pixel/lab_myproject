#!/usr/bin/env python3
"""
perception_6d_node_fixed.py

Fixed version: ensure PoseStamped in 'world' frame matches absolute Gazebo/world coordinates
produced by spawn_random_blocks.py.

Main fixes:
 - use the image message timestamp when transforming (keeps TF time consistent)
 - use tf.TransformListener.transformPose to transform the full PoseStamped (avoids manual
   quaternion/translation mistakes)
 - more robust fallback when stamp==0
 - small clarifying comments and logging to help debug mismatches with spawn_random_blocks

Requirements: camera_info and depth must be registered to the same image plane (depth_registered),
CameraInfo must be published, and TF must publish a transform between camera frame and world.
"""
import os
import rospy
import numpy as np
import cv2
import tf
import re
import time
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import message_filters

# Defaults
DEFAULT_MODEL_PATH = "/root/ros_ws/src/lab_myproject/data/runs/detect/train/weights/best.pt"
DEFAULT_COLOR_TOPIC = "/ur5/zed_node/left/image_rect_color"
DEFAULT_DEPTH_TOPIC = "/ur5/zed_node/depth/depth_registered"
DEFAULT_CAMINFO_TOPIC = "/ur5/zed_node/left_raw/camera_info"
DEFAULT_FRAME_CAMERA = "zed2_left_camera_optical_frame"
DEFAULT_FRAME_WORLD = "world"
DEFAULT_DEBUG_DIR = "/tmp/perception_debug"

DEFAULT_CONF_THRESH = 0.25
DEFAULT_MIN_POINTS_FOR_PCA = 5
DEFAULT_MIN_POINTS_FOR_PCA_3D = 30

# Depth valid range (meters)
DEPTH_MIN = 0.02
DEPTH_MAX = 5.0

# Helper utilities
def safe_name(name: str) -> str:
    s = re.sub(r'[^0-9a-zA-Z_]', '_', name)
    s = re.sub(r'__+', '_', s)
    return s.lower()


def compute_pca_components(points, n_components=3):
    """Compute principal components using SVD (rows = components)."""
    if points.shape[0] == 0:
        return np.zeros((n_components, points.shape[1]))
    mean = np.mean(points, axis=0)
    X = points - mean
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        comps = Vt[:n_components]
        return comps
    except Exception:
        D = points.shape[1]
        return np.eye(D)[:n_components]


class Perception6DNode:
    def __init__(self):
        rospy.init_node("perception_6d_node", anonymous=True)

        # params
        self.model_path = rospy.get_param("~model_path", DEFAULT_MODEL_PATH)
        self.color_topic = rospy.get_param("~color_topic", DEFAULT_COLOR_TOPIC)
        self.depth_topic = rospy.get_param("~depth_topic", DEFAULT_DEPTH_TOPIC)
        self.caminfo_topic = rospy.get_param("~caminfo_topic", DEFAULT_CAMINFO_TOPIC)
        self.frame_camera = rospy.get_param("~frame_camera", DEFAULT_FRAME_CAMERA)
        self.frame_world = rospy.get_param("~frame_world", DEFAULT_FRAME_WORLD)
        self.debug_dir = rospy.get_param("~debug_dir", DEFAULT_DEBUG_DIR)

        self.conf_thresh = float(rospy.get_param("~conf_thresh", DEFAULT_CONF_THRESH))
        self.min_points_pca = int(rospy.get_param("~min_points_for_pca", DEFAULT_MIN_POINTS_FOR_PCA))
        self.min_points_pca_3d = int(rospy.get_param("~min_points_for_pca_3d", DEFAULT_MIN_POINTS_FOR_PCA_3D))

        os.makedirs(self.debug_dir, exist_ok=True)

        rospy.loginfo(f"[perception6d] model={self.model_path}")
        rospy.loginfo(f"[perception6d] color={self.color_topic} depth={self.depth_topic} caminfo={self.caminfo_topic}")
        rospy.loginfo(f"[perception6d] camera_frame={self.frame_camera} world_frame={self.frame_world}")
        rospy.loginfo(f"[perception6d] conf={self.conf_thresh} pca_min={self.min_points_pca}/{self.min_points_pca_3d}")

        # load YOLO
        try:
            self.model = YOLO(self.model_path)
            rospy.loginfo("[perception6d] YOLO model loaded")
        except Exception as e:
            rospy.logerr(f"[perception6d] Failed to load YOLO model: {e}")
            raise

        # cv bridge and tf
        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()
        self.cam_K = None

        # publishers
        self.debug_image_pub = rospy.Publisher("/perception/debug/image_raw", Image, queue_size=1)
        self.pose_pubs = {}  # topic -> publisher

        # subscribe CameraInfo
        rospy.Subscriber(self.caminfo_topic, CameraInfo, self._caminfo_cb, queue_size=1)

        # sync color+depth
        color_sub = message_filters.Subscriber(self.color_topic, Image, queue_size=1, buff_size=2**24)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image, queue_size=1, buff_size=2**24)
        ats = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=6, slop=0.15)
        ats.registerCallback(self._synced_cb)

        rospy.loginfo("[perception6d] initialized — waiting for data")

    def _caminfo_cb(self, msg: CameraInfo):
        if self.cam_K is None:
            try:
                K = np.array(msg.K).reshape(3,3)
                self.cam_K = {'fx': float(K[0,0]), 'fy': float(K[1,1]), 'cx': float(K[0,2]), 'cy': float(K[1,2])}
                rospy.loginfo(f"[perception6d] camera intrinsics: {self.cam_K}")
            except Exception as e:
                rospy.logwarn(f"[perception6d] CameraInfo parse failed: {e}")

    def _try_expand_and_get_depth(self, depth_cv, x1c, y1c, x2c, y2c, expand_steps=2):
        """
        Try to get depth points inside bbox; if none, expand bbox slightly up to `expand_steps` times.
        Returns (us, vs, zs) arrays (may be empty).
        """
        h, w = depth_cv.shape[:2]
        for step in range(expand_steps + 1):
            # expansion factor: 1.0 + 0.2*step
            factor = 1.0 + 0.2 * step
            cx = (x1c + x2c) / 2.0
            cy = (y1c + y2c) / 2.0
            half_w = (x2c - x1c) / 2.0 * factor
            half_h = (y2c - y1c) / 2.0 * factor
            nx1 = int(max(0, np.floor(cx - half_w)))
            nx2 = int(min(w-1, np.ceil(cx + half_w)))
            ny1 = int(max(0, np.floor(cy - half_h)))
            ny2 = int(min(h-1, np.ceil(cy + half_h)))
            crop = depth_cv[ny1:ny2+1, nx1:nx2+1]
            if crop.size == 0:
                continue
            mask = np.isfinite(crop) & (crop > DEPTH_MIN) & (crop < DEPTH_MAX)
            ys, xs = np.where(mask)
            if xs.size > 0:
                zs = crop[ys, xs].astype(np.float32)
                # convert local crop coords to image coords
                us = (nx1 + xs).astype(np.float32)
                vs = (ny1 + ys).astype(np.float32)
                return us, vs, zs
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    def _synced_cb(self, color_msg: Image, depth_msg: Image):
        if self.cam_K is None:
            rospy.logwarn_throttle(5, "[perception6d] waiting for camera_info")
            return

        try:
            color_cv = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn(f"[perception6d] cvbridge color error: {e}")
            return

        # depth conversion
        try:
            enc = depth_msg.encoding
            if enc == "32FC1":
                depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)
            elif enc == "16UC1":
                depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32) / 1000.0
            else:
                depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)
        except Exception as e:
            rospy.logwarn(f"[perception6d] depth conversion failed: {e}")
            return

        # run YOLO inference
        try:
            results = self.model.predict(source=color_cv, conf=self.conf_thresh, imgsz=640, verbose=False)
            if len(results) == 0:
                return
            res = results[0]
        except Exception as e:
            rospy.logwarn(f"[perception6d] YOLO inference failed: {e}")
            return

        boxes = getattr(res.boxes, "xyxy", None)
        names = getattr(res, "names", {})
        if boxes is None:
            return

        fx = self.cam_K['fx']; fy = self.cam_K['fy']; cx = self.cam_K['cx']; cy = self.cam_K['cy']
        debug_img = color_cv.copy()

        # choose a transform timestamp consistent with the image
        stamp = color_msg.header.stamp if color_msg.header.stamp != rospy.Time(0) else rospy.Time.now()

        for i, b in enumerate(boxes):
            try:
                x1, y1, x2, y2 = int(b[0].item()), int(b[1].item()), int(b[2].item()), int(b[3].item())
            except Exception:
                continue

            

            try:
                class_id = int(res.boxes.cls[i].item())
                class_name = names[class_id] if class_id in names else str(class_id)
            except Exception:
                class_name = "obj"

            # try to obtain depth points (with small bbox expansion if needed)
            us, vs, zs = self._try_expand_and_get_depth(depth_cv, x1, y1, x2, y2, expand_steps=2)
            if us.size == 0:
                # no depth available — mark yellow bbox and continue (skip publishing pose)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # yellow
                cv2.putText(debug_img, class_name + " (no depth)", (x1, max(y1-10,0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                rospy.logdebug(f"[perception6d] no depth for {class_name}, skipping pose publish")
                continue

            # back-project to camera frame
            X = (us - cx) * zs / fx
            Y = (vs - cy) * zs / fy
            Z = zs
            pts = np.vstack((X, Y, Z)).T  # N x 3

            # centroid in camera frame
            centroid = np.nanmedian(pts, axis=0)

            # orientation: PCA3D if enough points, else PCA2D for yaw
            q_obj_cam = None
            roll = pitch = yaw = 0.0
            if pts.shape[0] >= self.min_points_pca_3d:
                try:
                    comps = compute_pca_components(pts, n_components=3)
                    x_axis = comps[0].copy(); y_axis = comps[1].copy(); z_axis = comps[2].copy()
                    if np.dot(np.cross(x_axis, y_axis), z_axis) < 0:
                        z_axis = -z_axis
                    R = np.column_stack((x_axis, y_axis, z_axis))
                    U, _, Vt = np.linalg.svd(R)
                    R_orth = np.dot(U, Vt)
                    if np.linalg.det(R_orth) < 0:
                        U[:, -1] *= -1
                        R_orth = np.dot(U, Vt)
                    R4 = np.eye(4, dtype=float); R4[:3, :3] = R_orth
                    q_obj_cam = tf.transformations.quaternion_from_matrix(R4)
                    roll, pitch, yaw = tf.transformations.euler_from_matrix(R4, axes='sxyz')
                except Exception as e:
                    rospy.logwarn_throttle(10, f"[perception6d] PCA3D failed: {e}")
                    q_obj_cam = None

            if q_obj_cam is None:
                # PCA 2D fallback for yaw
                try:
                    xy = pts[:, :2]
                    if xy.shape[0] >= max(2, self.min_points_pca):
                        comps2 = compute_pca_components(xy, n_components=2)
                        vec = comps2[0]
                        yaw = np.arctan2(vec[1], vec[0])
                    else:
                        yaw = 0.0
                except Exception:
                    yaw = 0.0
                q_obj_cam = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
                roll = pitch = 0.0

            # pose in camera frame
            tx_cam, ty_cam, tz_cam = float(centroid[0]), float(centroid[1]), float(centroid[2])
            pose_cam = PoseStamped()
            pose_cam.header.frame_id = self.frame_camera
            pose_cam.header.stamp = stamp
            pose_cam.pose.position.x = tx_cam
            pose_cam.pose.position.y = ty_cam
            pose_cam.pose.position.z = tz_cam
            pose_cam.pose.orientation.x = float(q_obj_cam[0])
            pose_cam.pose.orientation.y = float(q_obj_cam[1])
            pose_cam.pose.orientation.z = float(q_obj_cam[2])
            pose_cam.pose.orientation.w = float(q_obj_cam[3])

            # transform to world frame using transformPose (handles time & rotation properly)
            try:
                # wait for transform at the same stamp as the image (or reasonable fallback)
                self.tf_listener.waitForTransform(self.frame_world, self.frame_camera, stamp, rospy.Duration(1.0))
                pose_world = self.tf_listener.transformPose(self.frame_world, pose_cam)
                # Ensure world header stamp is also set to image stamp (useful for consumers)
                pose_world.header.stamp = stamp
            except Exception as e:
                rospy.logwarn_throttle(5, f"[perception6d] TF transform to world failed: {e}")
                # fallback: publish camera-frame pose but label clearly
                pose_world = pose_cam
                pose_world.header.frame_id = self.frame_camera

            # FILTER by world Z-range and Y>=0 in world frame
            z_world = None
            y_world = None
            try:
                if pose_world.header.frame_id == self.frame_world:
                    z_world = float(pose_world.pose.position.z)
                    y_world = float(pose_world.pose.position.y)
            except Exception:
                z_world = None
                y_world = None

            # reject if Z outside table band
            if z_world is not None and (z_world < 0.60 or z_world > 0.80):
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0,0,255), 2)  # red
                cv2.putText(debug_img, f"{class_name} (z out)", (x1, max(y1-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                rospy.loginfo(f"[perception6d] {class_name} ignored due to z_out_of_range: z={z_world:.3f}")
                continue

            # reject if Y is negative in world frame
            if y_world is not None and y_world < 0.0:
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0,0,255), 2)  # red
                cv2.putText(debug_img, f"{class_name} (y<0)", (x1, max(y1-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                rospy.loginfo(f"[perception6d] {class_name} ignored due to y_negative: y={y_world:.3f}")
                continue

            # publish PoseStamped on world-named topic
            topic = f"/detected_object_pose_{safe_name(class_name)}_world"
            if topic not in self.pose_pubs:
                self.pose_pubs[topic] = rospy.Publisher(topic, PoseStamped, queue_size=5)
                rospy.loginfo(f"[perception6d] publishing poses on {topic}")
            self.pose_pubs[topic].publish(pose_world)
            topic = f"/detected_object_pose_{safe_name(class_name)}_world"
            if topic not in self.pose_pubs:
                self.pose_pubs[topic] = rospy.Publisher(topic, PoseStamped, queue_size=5)
                rospy.loginfo(f"[perception6d] publishing poses on {topic}")
            self.pose_pubs[topic].publish(pose_world)

            # draw accepted detection (green)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0,255,0), 2)
            # show class + z(world) + yaw degrees
            try:
                yaw_world = tf.transformations.euler_from_quaternion([pose_world.pose.orientation.x,
                                                                      pose_world.pose.orientation.y,
                                                                      pose_world.pose.orientation.z,
                                                                      pose_world.pose.orientation.w])[2]
                yaw_deg = np.degrees(yaw_world)
            except Exception:
                yaw_deg = 0.0
            label = f"{class_name} z={pose_world.pose.position.z:.3f} y={yaw_deg:.1f}deg"
            cv2.putText(debug_img, label, (x1, max(y1-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # log a concise line
            rospy.loginfo(f"[perception6d] {class_name} -> world: x={pose_world.pose.position.x:.3f} y={pose_world.pose.position.y:.3f} z={pose_world.pose.position.z:.3f} yaw={yaw_deg:.1f}deg")

        # publish debug image
        try:
            img_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            img_msg.header.stamp = stamp
            img_msg.header.frame_id = self.frame_camera
            self.debug_image_pub.publish(img_msg)
        except Exception as e:
            rospy.logwarn_throttle(20, f"[perception6d] debug image publish failed: {e}")

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = Perception6DNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
