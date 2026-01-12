#!/usr/bin/env python3
"""
perception_6d_node.py (with roll/pitch estimation)

Same as previous node but now computes full 6DOF orientation (roll, pitch, yaw)
by applying PCA on the 3D points inside the detection bbox. Falls back to yaw-only
if insufficient points are available.

Inputs:
 - color:  /camera/color/image_raw
 - depth:  /camera/depth/image_rect_raw
 - caminfo: /camera/color/camera_info

Outputs:
 - PoseStamped per classe on /detected_object_pose_<class>
 - debug images in DEBUG_DIR
"""
import os
import rospy
import numpy as np
import cv2
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf
from ultralytics import YOLO
from sklearn.decomposition import PCA
from std_msgs.msg import Header

# ---------- Parametri configurabili ----------
MODEL_PATH = rospy.get_param("~model_path", "runs/detect/train/weights/best.pt")
COLOR_TOPIC = rospy.get_param("~color_topic", "/camera/color/image_raw")
DEPTH_TOPIC = rospy.get_param("~depth_topic", "/camera/depth/image_rect_raw")
CAMINFO_TOPIC = rospy.get_param("~caminfo_topic", "/camera/color/camera_info")
FRAME_CAMERA = rospy.get_param("~frame_camera", "camera_color_frame")
FRAME_BASE = rospy.get_param("~frame_base", "base_link")
MIN_POINTS_FOR_PCA = rospy.get_param("~min_points_for_pca", 30)        # for XY-yaw fallback
MIN_POINTS_FOR_PCA_3D = rospy.get_param("~min_points_for_pca_3d", 100) # for full 3D PCA
DEBUG_DIR = rospy.get_param("~debug_dir", "/tmp/perception_debug")
CONF_THRESH = rospy.get_param("~conf_thresh", 0.25)
os.makedirs(DEBUG_DIR, exist_ok=True)
# --------------------------------------------

class Perception6DNode:
    def __init__(self):
        rospy.init_node("perception_6d_node", anonymous=True)

        self.bridge = CvBridge()
        self.model = YOLO(MODEL_PATH)
        rospy.loginfo(f"Loaded YOLO model: {MODEL_PATH}")

        self.cam_K = None    # intrinsics fx,fy,cx,cy
        self.latest_depth = None
        self.latest_color = None

        rospy.Subscriber(CAMINFO_TOPIC, CameraInfo, self.caminfo_cb, queue_size=1)
        rospy.Subscriber(COLOR_TOPIC, Image, self.color_cb, queue_size=1)
        rospy.Subscriber(DEPTH_TOPIC, Image, self.depth_cb, queue_size=1)

        self.tf_listener = tf.TransformListener()
        self.pose_pubs = {}
        rospy.loginfo("Perception6DNode ready (with full 6DOF estimation).")

    def caminfo_cb(self, msg: CameraInfo):
        if self.cam_K is None:
            K = np.array(msg.K).reshape(3,3)
            self.cam_K = {'fx': float(K[0,0]), 'fy': float(K[1,1]), 'cx': float(K[0,2]), 'cy': float(K[1,2])}
            rospy.loginfo(f"Camera intrinsics: {self.cam_K}")

    def color_cb(self, msg: Image):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn(f"color_cb error: {e}")
            self.latest_color = None
        if self.latest_depth is not None and self.cam_K is not None:
            self.process_frame(msg.header)

    def depth_cb(self, msg: Image):
        try:
            if msg.encoding == "32FC1":
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            else:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
                if msg.encoding == "16UC1":
                    depth = depth / 1000.0
            self.latest_depth = depth
        except Exception as e:
            rospy.logwarn(f"depth_cb error: {e}")
            self.latest_depth = None

    def process_frame(self, header: Header):
        color = self.latest_color.copy()
        depth = self.latest_depth
        if color is None or depth is None or self.cam_K is None:
            return

        results = self.model.predict(source=color, conf=CONF_THRESH, imgsz=640, verbose=False)
        if len(results) == 0:
            return
        res = results[0]

        boxes = getattr(res.boxes, "xyxy", None)
        if boxes is None:
            rospy.loginfo("No boxes in frame.")
            return

        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = map(int, [b[0].item(), b[1].item(), b[2].item(), b[3].item()])
            class_id = int(res.boxes.cls[i].item())
            class_name = res.names[class_id] if class_id in res.names else str(class_id)

            # crop depth region
            h, w = depth.shape[:2]
            x1c = max(0, x1); y1c = max(0, y1); x2c = min(w-1, x2); y2c = min(h-1, y2)
            crop = depth[y1c:y2c+1, x1c:x2c+1]
            if crop.size == 0:
                rospy.logwarn("Empty crop for bbox, skipping.")
                continue

            # mask valid depths
            mask = np.isfinite(crop) & (crop > 0.01) & (crop < 5.0)
            ys, xs = np.where(mask)
            if xs.size < 5:
                rospy.logwarn(f"Too few valid points ({xs.size}) for object {class_name}, skipping.")
                continue

            zs = crop[ys, xs].astype(np.float32)
            us = (x1c + xs).astype(np.float32)
            vs = (y1c + ys).astype(np.float32)

            fx = self.cam_K['fx']; fy = self.cam_K['fy']; cx = self.cam_K['cx']; cy = self.cam_K['cy']
            X = (us - cx) * zs / fx
            Y = (vs - cy) * zs / fy
            Z = zs
            pts = np.vstack((X, Y, Z)).T  # N x 3 in camera frame

            # centroid robusto
            centroid = np.nanmedian(pts, axis=0)

            # Decide whether to compute full 3D PCA
            q_obj_cam = None
            roll = pitch = yaw = 0.0

            if pts.shape[0] >= MIN_POINTS_FOR_PCA_3D:
                try:
                    pca3 = PCA(n_components=3)
                    pca3.fit(pts)
                    comps = pca3.components_  # shape (3,3): comps[0]=pc1,...
                    x_axis = comps[0].copy()
                    y_axis = comps[1].copy()
                    z_axis = comps[2].copy()
                    # ensure right-handed coordinate system
                    if np.dot(np.cross(x_axis, y_axis), z_axis) < 0:
                        z_axis = -z_axis
                    # re-orthonormalize (just to be safe) using SVD
                    R = np.column_stack((x_axis, y_axis, z_axis))
                    # orthonormalize via SVD: R = U * Vt
                    U, _, Vt = np.linalg.svd(R)
                    R_orth = np.dot(U, Vt)
                    # ensure det = +1
                    if np.linalg.det(R_orth) < 0:
                        U[:, -1] *= -1
                        R_orth = np.dot(U, Vt)
                    R_mat = R_orth  # 3x3 rotation: object axes expressed in camera frame

                    # build 4x4 matrix for quaternion extraction
                    R4 = np.eye(4, dtype=float)
                    R4[:3, :3] = R_mat
                    q_obj_cam = tf.transformations.quaternion_from_matrix(R4)
                    # euler (roll, pitch, yaw) in camera frame
                    roll, pitch, yaw = tf.transformations.euler_from_matrix(R4, axes='sxyz')
                except Exception as e:
                    rospy.logwarn(f"PCA3D error for {class_name}: {e}")
                    q_obj_cam = None

            # fallback: if not computed, try PCA on XY to get yaw (old behavior)
            if q_obj_cam is None:
                try:
                    xy = pts[:, :2]
                    if xy.shape[0] >= MIN_POINTS_FOR_PCA:
                        pca2 = PCA(n_components=2)
                        pca2.fit(xy)
                        vec = pca2.components_[0]
                        yaw = np.arctan2(vec[1], vec[0])
                    else:
                        yaw = 0.0
                    # quaternion from yaw only (roll=pitch=0)
                    q_obj_cam = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
                    roll = pitch = 0.0
                except Exception as e:
                    rospy.logwarn(f"PCA2D fallback error: {e}")
                    q_obj_cam = tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0)
                    roll = pitch = yaw = 0.0

            # pose in camera frame
            tx_cam, ty_cam, tz_cam = float(centroid[0]), float(centroid[1]), float(centroid[2])
            pose_cam = PoseStamped()
            pose_cam.header.frame_id = FRAME_CAMERA
            pose_cam.header.stamp = rospy.Time.now()
            pose_cam.pose.position.x = tx_cam
            pose_cam.pose.position.y = ty_cam
            pose_cam.pose.position.z = tz_cam
            pose_cam.pose.orientation.x = float(q_obj_cam[0])
            pose_cam.pose.orientation.y = float(q_obj_cam[1])
            pose_cam.pose.orientation.z = float(q_obj_cam[2])
            pose_cam.pose.orientation.w = float(q_obj_cam[3])

            # transform to base frame if possible
            try:
                self.tf_listener.waitForTransform(FRAME_BASE, FRAME_CAMERA, rospy.Time(0), rospy.Duration(1.0))
                (trans, rot) = self.tf_listener.lookupTransform(FRAME_BASE, FRAME_CAMERA, rospy.Time(0))
                mat_cam_to_base = tf.transformations.concatenate_matrices(
                    tf.transformations.translation_matrix(trans),
                    tf.transformations.quaternion_matrix(rot)
                )
                p_cam = np.array([tx_cam, ty_cam, tz_cam, 1.0])
                p_base = np.dot(mat_cam_to_base, p_cam)
                q_base = tf.transformations.quaternion_multiply(rot, q_obj_cam)
                pose_base = PoseStamped()
                pose_base.header = pose_cam.header
                pose_base.header.frame_id = FRAME_BASE
                pose_base.pose.position.x = float(p_base[0])
                pose_base.pose.position.y = float(p_base[1])
                pose_base.pose.position.z = float(p_base[2])
                pose_base.pose.orientation.x = float(q_base[0])
                pose_base.pose.orientation.y = float(q_base[1])
                pose_base.pose.orientation.z = float(q_base[2])
                pose_base.pose.orientation.w = float(q_base[3])
            except Exception as e:
                rospy.logwarn(f"TF transform failed: {e}")
                pose_base = pose_cam

            # publish
            topic = f"/detected_object_pose_{class_name}"
            if topic not in self.pose_pubs:
                self.pose_pubs[topic] = rospy.Publisher(topic, PoseStamped, queue_size=5)
            self.pose_pubs[topic].publish(pose_base)

            # debug image and logs
            roll_deg = np.degrees(roll)
            pitch_deg = np.degrees(pitch)
            yaw_deg = np.degrees(yaw)
            rospy.loginfo(f"Published {class_name}: pos(base) = ({pose_base.pose.position.x:.3f}, {pose_base.pose.position.y:.3f}, {pose_base.pose.position.z:.3f}), roll={roll_deg:.1f}°, pitch={pitch_deg:.1f}°, yaw={yaw_deg:.1f}°")

            debug_img = color.copy()
            cv2.rectangle(debug_img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(debug_img, f"{class_name} r={roll_deg:.1f} p={pitch_deg:.1f} y={yaw_deg:.1f}", (x1, max(y1-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            fname = os.path.join(DEBUG_DIR, f"det_{class_name}_{rospy.Time.now().to_sec():.0f}.jpg")
            cv2.imwrite(fname, debug_img)

    def spin(self):
        rospy.spin()

def main():
    node = Perception6DNode()
    node.spin()

if __name__ == "__main__":
    main()
