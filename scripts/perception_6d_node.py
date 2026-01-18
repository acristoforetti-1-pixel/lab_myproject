#!/usr/bin/env python3
"""perception_6d_node.py (formatted + light fixes)

Minimal node that:
 - subscribes synchronized color + depth
 - runs YOLO on color
 - for every detection back-projects depth crop to 3D
 - computes centroid and orientation (PCA via SVD)
 - single height filter (base_link z in [z_min_base, z_max_base])
 - publishes PoseStamped on /detected_object_pose_<safe_name>
 - publishes debug image on /perception/debug/image_raw

Notes / changes compared to the provided snippet:
 - fixed a few syntax / indentation issues
 - made the depth encoding handling more robust
 - guarded access to YOLO result fields and names
 - cleaned up variable scoping and logging
 - added a couple of small defensive checks

This is intentionally conservative: it keeps the original approach
but is easier to read and a little more robust. Consider moving to
`tf2_ros` and `geometry_msgs/TransformStamped` in a follow-up.
"""

import os
import re
import time

import cv2
import numpy as np
import rospy
import tf
import message_filters
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO

# ---------------------------
# Defaults
# ---------------------------
DEFAULT_MODEL_PATH = (
    "/root/ros_ws/src/lab_myproject/data/runs/detect/train/weights/best.pt"
)
DEFAULT_COLOR_TOPIC = "/ur5/zed_node/left/image_rect_color"
DEFAULT_DEPTH_TOPIC = "/ur5/zed_node/depth/depth_registered"
DEFAULT_CAMINFO_TOPIC = "/ur5/zed_node/left_raw/camera_info"
DEFAULT_FRAME_CAMERA = "zed2_left_camera_optical_frame"
DEFAULT_FRAME_BASE = "base_link"
DEFAULT_DEBUG_DIR = "/tmp/perception_debug"
DEFAULT_CONF_THRESH = 0.25
DEFAULT_MIN_POINTS_FOR_PCA = 5
DEFAULT_MIN_POINTS_FOR_PCA_3D = 30
# table spawn defaults (from your spawn_random_blocks.py)
DEFAULT_TABLE_Z_WORLD = 0.82
DEFAULT_SPAWN_Z_OFFSET = 0.03
DEFAULT_Z_MARGIN = 0.50  # permissive by default

# depth clamps
DEPTH_MIN = 0.02
DEPTH_MAX = 5.0

# ---------------------------
# Helpers
# ---------------------------

def safe_name(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    s = re.sub(r"__+", "_", s)
    return s.lower()


def compute_pca_components(points: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Return the first n_components principal directions (rows) from SVD.

    points: (N, D) array
    returns: (n_components, D)
    """
    if points is None or points.size == 0:
        # if we don't know D, default to identity for 3D
        D = 3
        return np.eye(D)[:n_components]

    mean = np.mean(points, axis=0)
    X = points - mean
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        comps = Vt[:n_components]
        return comps
    except Exception:
        D = points.shape[1]
        return np.eye(D)[:n_components]


# ---------------------------
# Node
# ---------------------------
class Perception6DNode:
    def __init__(self):
        rospy.init_node("perception_6d_node", anonymous=True)

        # params
        self.model_path = rospy.get_param("~model_path", DEFAULT_MODEL_PATH)
        self.color_topic = rospy.get_param("~color_topic", DEFAULT_COLOR_TOPIC)
        self.depth_topic = rospy.get_param("~depth_topic", DEFAULT_DEPTH_TOPIC)
        self.caminfo_topic = rospy.get_param("~caminfo_topic", DEFAULT_CAMINFO_TOPIC)
        self.frame_camera = rospy.get_param("~frame_camera", DEFAULT_FRAME_CAMERA)
        self.frame_base = rospy.get_param("~frame_base", DEFAULT_FRAME_BASE)
        self.debug_dir = rospy.get_param("~debug_dir", DEFAULT_DEBUG_DIR)
        self.conf_thresh = float(rospy.get_param("~conf_thresh", DEFAULT_CONF_THRESH))
        self.min_points_pca = int(
            rospy.get_param("~min_points_for_pca", DEFAULT_MIN_POINTS_FOR_PCA)
        )
        self.min_points_pca_3d = int(
            rospy.get_param("~min_points_for_pca_3d", DEFAULT_MIN_POINTS_FOR_PCA_3D)
        )

        # table-based z filter
        self.table_z_world = float(rospy.get_param("~table_z_world", DEFAULT_TABLE_Z_WORLD))
        self.spawn_z_offset = float(rospy.get_param("~spawn_z_offset", DEFAULT_SPAWN_Z_OFFSET))
        self.z_margin = float(rospy.get_param("~z_margin", DEFAULT_Z_MARGIN))

        os.makedirs(self.debug_dir, exist_ok=True)

        rospy.loginfo(f"[perception6d] model={self.model_path}")
        rospy.loginfo(
            f"[perception6d] topics color={self.color_topic} depth={self.depth_topic} caminfo={self.caminfo_topic}"
        )
        rospy.loginfo(f"[perception6d] frames camera={self.frame_camera} base={self.frame_base}")
        rospy.loginfo(f"[perception6d] conf_thresh={self.conf_thresh} pca_min={self.min_points_pca}/{self.min_points_pca_3d}")
        rospy.loginfo(
            f"[perception6d] table_z_world={self.table_z_world} spawn_offset={self.spawn_z_offset} z_margin={self.z_margin}"
        )

        # load YOLO
        try:
            self.model = YOLO(self.model_path)
            rospy.loginfo("[perception6d] YOLO model loaded")
        except Exception as e:
            rospy.logerr(f"[perception6d] YOLO load failed: {e}")
            raise

        self.bridge = CvBridge()
        self.cam_K = None
        self.tf_listener = tf.TransformListener()

        # pose publishers map
        self.pose_pubs = {}

        # debug image publisher
        self.debug_image_pub = rospy.Publisher("/perception/debug/image_raw", Image, queue_size=1)

        # compute expected z range in base_link
        self.z_min_base = None
        self.z_max_base = None
        self._compute_z_range_from_world()

        # subscribe camera info (one-shot)
        rospy.Subscriber(self.caminfo_topic, CameraInfo, self._caminfo_cb, queue_size=1)

        # sync color+depth (approx)
        color_sub = message_filters.Subscriber(self.color_topic, Image, queue_size=1, buff_size=2 ** 24)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image, queue_size=1, buff_size=2 ** 24)
        ats = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=6, slop=0.15)
        ats.registerCallback(self._synced_cb)

        rospy.loginfo("[perception6d] Node initialized and waiting for messages...")

    def _compute_z_range_from_world(self):
        target = self.frame_base
        source = "world"
        z_world = float(self.table_z_world) + float(self.spawn_z_offset)
        rospy.loginfo(f"[perception6d] computing expected base z from world z={z_world:.3f}")
        try:
            self.tf_listener.waitForTransform(target, source, rospy.Time(0), rospy.Duration(2.0))
            (trans, rot) = self.tf_listener.lookupTransform(target, source, rospy.Time(0))

            mat = tf.transformations.concatenate_matrices(
                tf.transformations.translation_matrix(trans), tf.transformations.quaternion_matrix(rot)
            )
            p_world = np.array([0.0, 0.0, z_world, 1.0])
            p_base = np.dot(mat, p_world)
            z_base = float(p_base[2])

            self.z_min_base = z_base - abs(self.z_margin)
            self.z_max_base = z_base + abs(self.z_margin)
            rospy.loginfo(
                f"[perception6d] base z range: [{self.z_min_base:.3f}, {self.z_max_base:.3f}] (z_base={z_base:.3f})"
            )
        except Exception as e:
            rospy.logwarn(f"[perception6d] TF world->base not available: {e}. Using fallback z range.")
            self.z_min_base = -1.5
            self.z_max_base = 0.5
            rospy.loginfo(f"[perception6d] fallback base z range: [{self.z_min_base}, {self.z_max_base}]")

    def _caminfo_cb(self, msg: CameraInfo):
        if self.cam_K is None:
            try:
                K = np.array(msg.K).reshape(3, 3)
                self.cam_K = {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])}
                rospy.loginfo(f"[perception6d] camera intrinsics: {self.cam_K}")
            except Exception as e:
                rospy.logwarn(f"[perception6d] failed parse CameraInfo: {e}")

    def _synced_cb(self, color_msg: Image, depth_msg: Image):
        # ensure intrinsics
        if self.cam_K is None:
            rospy.logwarn_throttle(5, "[perception6d] waiting for camera_info...")
            return

        # convert color
        try:
            color_cv = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn(f"[perception6d] cvbridge color error: {e}")
            return

        # depth conversion
        try:
            enc = getattr(depth_msg, "encoding", "")
            if enc == "32FC1":
                depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)
            elif enc == "16UC1":
                depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32) / 1000.0
            else:
                # fallback: try passthrough and hope for the best
                depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)
                rospy.logwarn_throttle(20, f"[perception6d] unhandled depth encoding {enc}, trying passthrough")
        except Exception as e:
            rospy.logwarn(f"[perception6d] depth conversion error: {e}")
            return

        # YOLO inference
        try:
            results = self.model.predict(source=color_cv, conf=self.conf_thresh, imgsz=640, verbose=False)
            if results is None or len(results) == 0:
                return
            res = results[0]
        except Exception as e:
            rospy.logwarn(f"[perception6d] YOLO inference error: {e}")
            return

        # boxes (N,4) in xyxy
        boxes = None
        try:
            boxes = getattr(res.boxes, "xyxy", None)
            if boxes is None:
                return
            # convert to numpy if it's a tensor-like object
            boxes = np.array(boxes)
        except Exception:
            return

        fx = self.cam_K["fx"]
        fy = self.cam_K["fy"]
        cx = self.cam_K["cx"]
        cy = self.cam_K["cy"]

        debug_img = color_cv.copy()
        h, w = depth_cv.shape[:2]

        for i, b in enumerate(boxes):
            try:
                x1, y1, x2, y2 = int(b[0].item()) if hasattr(b[0], "item") else int(b[0]), int(b[1].item()) if hasattr(b[1], "item") else int(b[1]), int(b[2].item()) if hasattr(b[2], "item") else int(b[2]), int(b[3].item()) if hasattr(b[3], "item") else int(b[3])
            except Exception:
                rospy.logdebug("[perception6d] invalid bbox format, skipping")
                continue

            # class name (best-effort)
            try:
                class_id = int(res.boxes.cls[i].item()) if hasattr(res.boxes.cls[i], "item") else int(res.boxes.cls[i])
                class_name = res.names[class_id] if hasattr(res, "names") and class_id in res.names else str(class_id)
            except Exception:
                class_name = "obj"

            # clamp bbox and crop depth
            x1c = max(0, min(w - 1, x1))
            x2c = max(0, min(w - 1, x2))
            y1c = max(0, min(h - 1, y1))
            y2c = max(0, min(h - 1, y2))
            if x2c <= x1c or y2c <= y1c:
                continue

            crop = depth_cv[y1c : (y2c + 1), x1c : (x2c + 1)]
            if crop.size == 0:
                continue

            mask = np.isfinite(crop) & (crop > DEPTH_MIN) & (crop < DEPTH_MAX)
            ys, xs = np.where(mask)
            if xs.size < 3:
                rospy.logdebug(f"[perception6d] too few depth points ({xs.size}) for {class_name}")
                # draw rejected bbox in red
                cv2.rectangle(debug_img, (x1c, y1c), (x2c, y2c), (0, 0, 255), 2)
                cv2.putText(debug_img, class_name, (x1c, max(y1c - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue

            zs = crop[ys, xs].astype(np.float32)
            us = (x1c + xs).astype(np.float32)
            vs = (y1c + ys).astype(np.float32)

            X = (us - cx) * zs / fx
            Y = (vs - cy) * zs / fy
            Z = zs
            pts = np.vstack((X, Y, Z)).T  # (N,3) in camera frame

            # centroid (camera frame)
            centroid = np.nanmedian(pts, axis=0)

            # orientation via PCA (SVD)
            q_obj_cam = None
            roll = pitch = yaw = 0.0

            if pts.shape[0] >= self.min_points_pca_3d:
                try:
                    comps = compute_pca_components(pts, n_components=3)
                    x_axis = comps[0].copy()
                    y_axis = comps[1].copy()
                    z_axis = comps[2].copy()

                    # ensure right-handed
                    if np.dot(np.cross(x_axis, y_axis), z_axis) < 0:
                        z_axis = -z_axis

                    R = np.column_stack((x_axis, y_axis, z_axis))
                    U, _, Vt = np.linalg.svd(R)
                    R_orth = np.dot(U, Vt)
                    if np.linalg.det(R_orth) < 0:
                        U[:, -1] *= -1
                        R_orth = np.dot(U, Vt)

                    R4 = np.eye(4, dtype=float)
                    R4[:3, :3] = R_orth
                    q_obj_cam = tf.transformations.quaternion_from_matrix(R4)
                    roll, pitch, yaw = tf.transformations.euler_from_matrix(R4, axes="sxyz")
                except Exception as e:
                    rospy.logwarn_throttle(10, f"[perception6d] PCA3D failed: {e}")
                    q_obj_cam = None

            # fallback: 2D PCA on XY to estimate yaw
            if q_obj_cam is None:
                try:
                    xy = pts[:, :2]
                    if xy.shape[0] >= self.min_points_pca:
                        comps2 = compute_pca_components(xy, n_components=2)
                        vec = comps2[0]
                        yaw = np.arctan2(vec[1], vec[0])
                    else:
                        yaw = 0.0
                except Exception:
                    yaw = 0.0
                q_obj_cam = tf.transformations.quaternion_from_euler(0.0, 0.0, float(yaw))
                roll = pitch = 0.0

            # pose in camera frame
            tx_cam, ty_cam, tz_cam = float(centroid[0]), float(centroid[1]), float(centroid[2])
            pose_cam = PoseStamped()
            pose_cam.header.frame_id = self.frame_camera
            pose_cam.header.stamp = rospy.Time.now()
            pose_cam.pose.position.x = tx_cam
            pose_cam.pose.position.y = ty_cam
            pose_cam.pose.position.z = tz_cam
            pose_cam.pose.orientation.x = float(q_obj_cam[0])
            pose_cam.pose.orientation.y = float(q_obj_cam[1])
            pose_cam.pose.orientation.z = float(q_obj_cam[2])
            pose_cam.pose.orientation.w = float(q_obj_cam[3])

            # transform to base frame
            try:
                self.tf_listener.waitForTransform(self.frame_base, self.frame_camera, rospy.Time(0), rospy.Duration(1.0))
                (trans, rot) = self.tf_listener.lookupTransform(self.frame_base, self.frame_camera, rospy.Time(0))

                mat_cam_to_base = tf.transformations.concatenate_matrices(
                    tf.transformations.translation_matrix(trans), tf.transformations.quaternion_matrix(rot)
                )
                p_cam = np.array([tx_cam, ty_cam, tz_cam, 1.0])
                p_base = np.dot(mat_cam_to_base, p_cam)

                # rotate quaternion: q_base = rot * q_obj_cam
                q_base = tf.transformations.quaternion_multiply(rot, q_obj_cam)

                pose_base = PoseStamped()
                pose_base.header = pose_cam.header
                pose_base.header.frame_id = self.frame_base
                pose_base.pose.position.x = float(p_base[0])
                pose_base.pose.position.y = float(p_base[1])
                pose_base.pose.position.z = float(p_base[2])
                pose_base.pose.orientation.x = float(q_base[0])
                pose_base.pose.orientation.y = float(q_base[1])
                pose_base.pose.orientation.z = float(q_base[2])
                pose_base.pose.orientation.w = float(q_base[3])
            except Exception as e:
                rospy.logwarn_throttle(10, f"[perception6d] TF transform failed: {e}")
                # fallback to camera frame (not ideal but safe)
                pose_base = pose_cam

            # height filter in base_link (only filter present)
            zbase = pose_base.pose.position.z

            # extra hard cutoff to remove robot arm detections
            # anything above this z is rejected unconditionally
            Z_ARM_CUTOFF = -0.85  # tightened cutoff to avoid rejecting real blocks around -1.0
            if zbase > Z_ARM_CUTOFF:
                cv2.rectangle(debug_img, (x1c, y1c), (x2c, y2c), (0, 0, 255), 2)
                cv2.putText(
                    debug_img,
                    class_name + f" z={zbase:.3f}",
                    (x1c, max(y1c - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                rospy.logdebug(f"[perception6d] rejected (arm cutoff) {class_name} z={zbase:.3f}")
                continue

            if not (self.z_min_base <= zbase <= self.z_max_base):
                # rejected by table height window
                cv2.rectangle(debug_img, (x1c, y1c), (x2c, y2c), (0, 0, 255), 2)
                cv2.putText(
                    debug_img,
                    class_name + f" z={zbase:.3f}",
                    (x1c, max(y1c - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                rospy.logdebug(f"[perception6d] rejected by height window {class_name} z={zbase:.3f}")
                continue

            # accepted: publish PoseStamped (ROS-safe topic)
            topic = f"/detected_object_pose_{safe_name(class_name)}"
            if topic not in self.pose_pubs:
                self.pose_pubs[topic] = rospy.Publisher(topic, PoseStamped, queue_size=5)
                rospy.loginfo(f"[perception6d] publishing poses on {topic}")

            self.pose_pubs[topic].publish(pose_base)

            # draw bbox & label accepted (green)
            cv2.rectangle(debug_img, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
            cv2.putText(
                debug_img,
                class_name + f" z={zbase:.3f}",
                (x1c, max(y1c - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # small log
            rospy.loginfo_throttle(
                5,
                f"[perception6d] {class_name}: base_pos = ({pose_base.pose.position.x:.3f}, {pose_base.pose.position.y:.3f}, {pose_base.pose.position.z:.3f}), yaw={np.degrees(yaw):.1f}",
            )

        # publish debug image
        try:
            msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.frame_camera
            self.debug_image_pub.publish(msg)
        except Exception as e:
            rospy.logwarn_throttle(20, f"[perception6d] failed publish debug image: {e}")

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = Perception6DNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
