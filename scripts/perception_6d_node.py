#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# limita thread (aiuta stabilità con torch/opencv)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
import hashlib
from collections import deque

import numpy as np
import rospy
import tf
import message_filters
import cv2

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from std_msgs.msg import String, Float64MultiArray, Bool


DEFAULT_MODEL_PATH = "/root/ros_ws/src/lab_myproject/data/runs/detect/train/weights/best.pt"
DEFAULT_COLOR_TOPIC = "/ur5/zed_node/left/image_rect_color"
DEFAULT_DEPTH_TOPIC = "/ur5/zed_node/depth/depth_registered"
DEFAULT_CAMINFO_TOPIC = "/ur5/zed_node/left/camera_info"

DEFAULT_FRAME_BASE = "base_link"
DEFAULT_CONF_THRESH = 0.25

DEPTH_MIN = 0.02
DEPTH_MAX = 5.0


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class Perception6DNode:
    """
    Perception robusto e veloce:
      - YOLO su color (overlay SEMPRE su /perception/debug/image_raw)
      - depth sampling nel bbox (con shrink)
      - stima z tavolo in base_link (istogramma)
      - selezione punti oggetto: sopra tavolo + top-surface (percentile dz) per XY
      - XY: densest-bin (2D) + refine median in r
      - yaw robusto:
          * usa punti OBJ completi (footprint) per yaw
          * minAreaRect -> yaw_long (asse lungo)
          * fallback PCA major axis -> yaw_long
          * output yaw_mode: "long" (default) o "short" (+90°)
          * yaw_tool_offset: offset fisso (rad) per allineare col tool reale
          * yaw SNAP opzionale (es: 90°) per evitare jitter/ambiguità
          * outlier reject in request-mode (yaw sporchi non rovinano la mediana)
      - request-mode:
          * NON resetta se arrivano request ripetute mentre pending
          * pubblica entro max_wait se ha min_frames (no attese infinite)
      - /vision publishers latch (task_planning non perde messaggi)
    """

    def __init__(self):
        rospy.init_node("perception_6d_node", anonymous=False)

        # riduce rogne threading cv2
        try:
            cv2.setNumThreads(0)
        except Exception:
            pass

        rospy.on_shutdown(self._on_shutdown)

        # ---- base params ----
        self.model_path = rospy.get_param("~model_path", DEFAULT_MODEL_PATH)
        self.color_topic = rospy.get_param("~color_topic", DEFAULT_COLOR_TOPIC)
        self.depth_topic = rospy.get_param("~depth_topic", DEFAULT_DEPTH_TOPIC)
        self.caminfo_topic = rospy.get_param("~caminfo_topic", DEFAULT_CAMINFO_TOPIC)
        self.frame_base = rospy.get_param("~frame_base", DEFAULT_FRAME_BASE)
        self.conf_thresh = float(rospy.get_param("~conf_thresh", DEFAULT_CONF_THRESH))

        # ---- request mode ----
        self.publish_on_request = bool(rospy.get_param("~publish_on_request", True))
        self.request_topic = rospy.get_param("~request_topic", "/vision/request_object")
        self.request_timeout_s = float(rospy.get_param("~request_timeout", 3.0))

        self.req_collect_N = int(rospy.get_param("~req_collect_N", 5))
        self.req_min_frames = int(rospy.get_param("~req_min_frames", 2))
        self.req_max_wait_s = float(rospy.get_param("~req_max_wait", 0.60))  # pubblica veloce
        self.req_cooldown_s = float(rospy.get_param("~req_cooldown", 0.25))  # ignora spam request True

        self.min_pub_period = float(rospy.get_param("~min_pub_period", 0.05))

        self._req_pending = not self.publish_on_request
        self._req_start_t = rospy.Time(0)
        self._req_deadline = rospy.Time(0)
        self._last_req_msg = rospy.Time(0)
        self._last_pub = rospy.Time(0)

        self._req_buf = deque(maxlen=max(1, self.req_collect_N))

        # ---- table ROI (DEPTH percentuali) ----
        self.table_roi_u0 = float(rospy.get_param("~table_roi_u0", 0.38))
        self.table_roi_u1 = float(rospy.get_param("~table_roi_u1", 0.62))
        self.table_roi_v0 = float(rospy.get_param("~table_roi_v0", 0.58))
        self.table_roi_v1 = float(rospy.get_param("~table_roi_v1", 0.68))

        # ---- table filters ----
        self.use_table_z = bool(rospy.get_param("~use_table_z", True))
        self.table_z_tol = float(rospy.get_param("~table_z_tol", 0.06))

        self.use_obj_range_filter = bool(rospy.get_param("~use_obj_range_filter", True))
        self.table_obj_gap = float(rospy.get_param("~table_obj_gap", 0.008))  # 8mm
        self.table_obj_max = float(rospy.get_param("~table_obj_max", 0.25))   # 25cm

        # ---- XY bounds (base_link) ----
        self.use_table_xy = bool(rospy.get_param("~use_table_xy", False))
        self.table_x_min = float(rospy.get_param("~table_x_min", -0.35))
        self.table_x_max = float(rospy.get_param("~table_x_max",  0.30))
        self.table_y_min = float(rospy.get_param("~table_y_min",  0.10))
        self.table_y_max = float(rospy.get_param("~table_y_max",  0.40))

        # ---- robot ROI (in COLOR) ----
        self.robot_roi_enable = bool(rospy.get_param("~robot_roi_enable", False))
        self.robot_roi_u0 = float(rospy.get_param("~robot_roi_u0", 0.00))
        self.robot_roi_u1 = float(rospy.get_param("~robot_roi_u1", 0.55))
        self.robot_roi_v0 = float(rospy.get_param("~robot_roi_v0", 0.00))
        self.robot_roi_v1 = float(rospy.get_param("~robot_roi_v1", 0.70))

        # ---- precision knobs ----
        self.bbox_shrink = float(rospy.get_param("~bbox_shrink", 0.12))
        self.xy_inlier_r = float(rospy.get_param("~xy_inlier_r", 0.040))
        self.top_keep_percentile = float(rospy.get_param("~top_keep_percentile", 70.0))
        self.dense_bin_size = float(rospy.get_param("~dense_bin_size", 0.008))

        # bias (base_link)
        self.grasp_bias_x = float(rospy.get_param("~grasp_bias_x", 0.0))
        self.grasp_bias_y = float(rospy.get_param("~grasp_bias_y", 0.0))

        # ---- yaw robusto ----
        self.estimate_yaw = bool(rospy.get_param("~estimate_yaw", True))

        # output:
        #  - "long"  -> yaw asse lungo (DEFAULT)
        #  - "short" -> yaw asse corto (= long + 90°)
        self.yaw_mode = str(rospy.get_param("~yaw_mode", "long")).strip().lower()
        if self.yaw_mode not in ("long", "short"):
            self.yaw_mode = "long"

        # offset tool fisso (radianti). Se vuoi 90°: 1.5708
        self.yaw_tool_offset = float(rospy.get_param("~yaw_tool_offset", 0.0))

        # SNAP yaw (riduce jitter / ambiguità) – default ON a 90°
        self.yaw_snap_enable = bool(rospy.get_param("~yaw_snap_enable", True))
        self.yaw_snap_step = float(rospy.get_param("~yaw_snap_step", math.pi / 2.0))  # 90°
        self.yaw_outlier_deg = float(rospy.get_param("~yaw_outlier_deg", 25.0))       # request-mode reject

        # minAreaRect (OBB) robusto
        self.yaw_use_rect = bool(rospy.get_param("~yaw_use_rect", True))
        self.yaw_rect_min_aspect = float(rospy.get_param("~yaw_rect_min_aspect", 1.07))
        self.yaw_rect_min_pts = int(rospy.get_param("~yaw_rect_min_pts", 120))

        # fallback PCA major
        self.yaw_use_pca = bool(rospy.get_param("~yaw_use_pca", True))
        self.yaw_min_anisotropy = float(rospy.get_param("~yaw_min_anisotropy", 1.10))
        self.yaw_pca_min_pts = int(rospy.get_param("~yaw_pca_min_pts", 140))

        # usa punti obj completi per yaw (consigliato)
        self.yaw_use_obj_points = bool(rospy.get_param("~yaw_use_obj_points", True))

        # smoothing leggero dello yaw (anti jitter) – ATTENZIONE: con snap spesso puoi spegnerlo
        self.yaw_smooth = bool(rospy.get_param("~yaw_smooth", False))
        self.yaw_smooth_alpha = float(rospy.get_param("~yaw_smooth_alpha", 0.35))
        self._yaw_last = None  # yaw filtrato (rad)

        # ---- debug ----
        self.debug_enable = bool(rospy.get_param("~debug_enable", True))
        self.debug_print_every = float(rospy.get_param("~debug_print_every", 1.0))
        self._last_print = rospy.Time(0)

        # ---- latch su /vision ----
        self.latch_vision = bool(rospy.get_param("~latch_vision", True))

        # ---- ROS ----
        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()

        self.caminfo = None
        self.K0 = None
        self._K_cache = {}  # (w,h) -> (fx,fy,cx,cy)

        self.model = YOLO(self.model_path)

        self.pub_obj_pose = rospy.Publisher("/vision/object_pose", PoseStamped, queue_size=1, latch=self.latch_vision)
        self.pub_obj_rpy  = rospy.Publisher("/vision/object_rpy",  Float64MultiArray, queue_size=1, latch=self.latch_vision)
        self.pub_obj_name = rospy.Publisher("/vision/object_name", String, queue_size=1, latch=self.latch_vision)
        self.pub_obj_uid  = rospy.Publisher("/vision/object_uid",  String, queue_size=1, latch=self.latch_vision)

        self.pub_debug = rospy.Publisher("/perception/debug/image_raw", Image, queue_size=1, latch=False)

        rospy.Subscriber(self.caminfo_topic, CameraInfo, self._caminfo_cb, queue_size=1)
        rospy.Subscriber(self.request_topic, Bool, self._req_cb, queue_size=3)

        color_sub = message_filters.Subscriber(self.color_topic, Image, queue_size=1, buff_size=2**24)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image, queue_size=1, buff_size=2**24)
        ats = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=6, slop=0.15)
        ats.registerCallback(self._synced_cb)

        self._best = None
        self._have_best = False

        rospy.loginfo("[perception6d] model=%s", self.model_path)
        rospy.loginfo("[perception6d] color=%s", self.color_topic)
        rospy.loginfo("[perception6d] depth=%s", self.depth_topic)
        rospy.loginfo("[perception6d] caminfo=%s", self.caminfo_topic)
        rospy.loginfo("[perception6d] base=%s conf=%.2f", self.frame_base, self.conf_thresh)

        rospy.logwarn("[perception6d] request=%s topic=%s timeout=%.1fs N=%d min=%d max_wait=%.2fs cooldown=%.2fs latch_vision=%s",
                      str(self.publish_on_request), self.request_topic, self.request_timeout_s,
                      self.req_collect_N, self.req_min_frames, self.req_max_wait_s, self.req_cooldown_s, str(self.latch_vision))

        rospy.logwarn("[perception6d] bbox_shrink=%.2f top%%=%.1f dense_bin=%.3f xy_r=%.3f bias=(%.3f,%.3f)",
                      self.bbox_shrink, self.top_keep_percentile, self.dense_bin_size, self.xy_inlier_r,
                      self.grasp_bias_x, self.grasp_bias_y)

        rospy.logwarn("[perception6d] yaw estimate=%s mode=%s tool_offset=%.3f snap=%s step=%.3f outlier=%.1fdeg rect=%s asp>=%.2f pca=%s anis>=%.2f use_obj=%s smooth=%s",
                      str(self.estimate_yaw), self.yaw_mode, self.yaw_tool_offset,
                      str(self.yaw_snap_enable), self.yaw_snap_step, self.yaw_outlier_deg,
                      str(self.yaw_use_rect), self.yaw_rect_min_aspect,
                      str(self.yaw_use_pca), self.yaw_min_anisotropy,
                      str(self.yaw_use_obj_points), str(self.yaw_smooth))

    def _on_shutdown(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            self.model = None
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ---------------- request ----------------
    def _req_cb(self, msg: Bool):
        if not self.publish_on_request:
            return
        if not msg.data:
            return

        now = rospy.Time.now()

        # ignora spam di request
        if (now - self._last_req_msg).to_sec() < self.req_cooldown_s:
            return
        self._last_req_msg = now

        # IMPORTANTISSIMO: se è già pending, NON resettare buffer
        if self._req_pending:
            return

        self._req_pending = True
        self._req_start_t = now
        self._req_deadline = now + rospy.Duration(self.request_timeout_s)
        self._req_buf.clear()

        # reset filtro yaw: evita trascinamento tra oggetti
        self._yaw_last = None

        rospy.loginfo("[perception6d] request received -> median over %d frames (min=%d, max_wait=%.2fs, timeout %.1fs)",
                      self.req_collect_N, self.req_min_frames, self.req_max_wait_s, self.request_timeout_s)

    # ---------------- cam info ----------------
    def _caminfo_cb(self, msg: CameraInfo):
        if self.caminfo is not None:
            return
        self.caminfo = msg
        K = np.array(msg.K, dtype=float).reshape(3, 3)
        self.K0 = {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "w": int(msg.width),
            "h": int(msg.height),
        }
        rospy.loginfo("[perception6d] caminfo fx=%.2f fy=%.2f cx=%.2f cy=%.2f (w=%d h=%d)",
                      self.K0["fx"], self.K0["fy"], self.K0["cx"], self.K0["cy"], self.K0["w"], self.K0["h"])

    # ---------------- utils ----------------
    @staticmethod
    def _draw_label(img, x1, y1, x2, y2, text, color=(0, 255, 0), thick=2):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bx1 = x1
        by1 = max(y1 - th - 10, 0)
        bx2 = min(x1 + tw + 8, img.shape[1] - 1)
        by2 = min(by1 + th + 10, img.shape[0] - 1)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.putText(img, text, (x1 + 4, by2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def _publish_debug(self, img_bgr, frame_id):
        if not self.debug_enable:
            return
        try:
            out = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
            out.header.stamp = rospy.Time.now()
            out.header.frame_id = frame_id
            self.pub_debug.publish(out)
        except Exception:
            pass

    def _depth_to_meters(self, depth_msg: Image):
        enc = getattr(depth_msg, "encoding", "")
        if enc == "32FC1":
            return self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)
        if enc == "16UC1":
            return (self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32) / 1000.0)
        return self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)

    def _tf_to_base(self, source_frame: str, stamp: rospy.Time):
        for use_stamp in (True, False):
            try:
                ts = stamp if use_stamp else rospy.Time(0)
                self.tf_listener.waitForTransform(self.frame_base, source_frame, ts, rospy.Duration(0.6))
                (trans, rot) = self.tf_listener.lookupTransform(self.frame_base, source_frame, ts)
                T = tf.transformations.concatenate_matrices(
                    tf.transformations.translation_matrix(trans),
                    tf.transformations.quaternion_matrix(rot),
                )
                return T
            except Exception:
                pass
        return None

    def _intrinsics_for_image(self, w_img: int, h_img: int):
        key = (int(w_img), int(h_img))
        if key in self._K_cache:
            return self._K_cache[key]

        fx0, fy0, cx0, cy0 = self.K0["fx"], self.K0["fy"], self.K0["cx"], self.K0["cy"]
        w0, h0 = float(self.K0["w"]), float(self.K0["h"])

        sx = float(w_img) / w0 if w0 > 0 else 1.0
        sy = float(h_img) / h0 if h0 > 0 else 1.0

        fx = fx0 * sx
        fy = fy0 * sy
        cx = cx0 * sx
        cy = cy0 * sy

        cx_exp = 0.5 * float(w_img)
        cy_exp = 0.5 * float(h_img)

        # fix principal point (tuo caso: cx=640 su 1920)
        if abs(cx - cx_exp) > 0.10 * w_img or abs(cy - cy_exp) > 0.10 * h_img:
            sx_c = cx_exp / max(cx, 1e-6)
            sy_c = cy_exp / max(cy, 1e-6)
            s = 0.5 * (sx_c + sy_c)
            fx *= s
            fy *= s
            cx *= s
            cy *= s
            rospy.logwarn_throttle(
                2.0,
                "[perception6d] K FIX: raw(cx,cy)=(%.1f,%.1f) expect=(%.1f,%.1f) -> s=%.3f new(cx,cy)=(%.1f,%.1f)",
                cx0, cy0, cx_exp, cy_exp, s, cx, cy
            )

        self._K_cache[key] = (fx, fy, cx, cy)
        return fx, fy, cx, cy

    @staticmethod
    def _sample_bbox_points(depth, x1, y1, x2, y2, max_pts=3200):
        crop = depth[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        m = np.isfinite(crop) & (crop > DEPTH_MIN) & (crop < DEPTH_MAX)
        ys, xs = np.where(m)
        if xs.size < 140:
            return None
        N = min(max_pts, xs.size)
        idx = np.random.choice(xs.size, size=N, replace=False)
        xs = xs[idx].astype(np.float32)
        ys = ys[idx].astype(np.float32)
        zs = crop[ys.astype(int), xs.astype(int)].astype(np.float32)
        us = (x1 + xs).astype(np.float32)
        vs = (y1 + ys).astype(np.float32)
        return us, vs, zs

    def _estimate_table_z_base(self, depth, T_base_cam, fx, fy, cx, cy):
        h, w = depth.shape[:2]
        u0 = int(np.clip(self.table_roi_u0 * w, 0, w - 1))
        u1 = int(np.clip(self.table_roi_u1 * w, 1, w))
        v0 = int(np.clip(self.table_roi_v0 * h, 0, h - 1))
        v1 = int(np.clip(self.table_roi_v1 * h, 1, h))
        if u1 <= u0 + 10 or v1 <= v0 + 10:
            return None, (u0, v0, u1, v1)

        roi = depth[v0:v1, u0:u1]
        m = np.isfinite(roi) & (roi > DEPTH_MIN) & (roi < DEPTH_MAX)
        if np.count_nonzero(m) < 500:
            return None, (u0, v0, u1, v1)

        ys, xs = np.where(m)
        N = min(3500, xs.size)
        idx = np.random.choice(xs.size, size=N, replace=False)
        xs = xs[idx].astype(np.float32)
        ys = ys[idx].astype(np.float32)
        zs = roi[ys.astype(int), xs.astype(int)].astype(np.float32)
        us = (u0 + xs).astype(np.float32)
        vs = (v0 + ys).astype(np.float32)

        X = (us - cx) * zs / fx
        Y = (vs - cy) * zs / fy
        Z = zs
        ones = np.ones_like(Z, dtype=np.float32)
        pts_cam = np.stack([X, Y, Z, ones], axis=0)
        pts_base = T_base_cam @ pts_cam
        z_base = pts_base[2, :]

        if z_base.size < 300:
            return None, (u0, v0, u1, v1)

        bin_size = 0.01
        zmin = float(np.nanmin(z_base))
        zmax = float(np.nanmax(z_base))
        nb = int(max(10, min(300, (zmax - zmin) / bin_size)))
        hist, edges = np.histogram(z_base, bins=nb)
        k = int(np.argmax(hist))
        z_peak = 0.5 * (edges[k] + edges[k + 1])

        win = 0.02
        sel = z_base[(z_base > (z_peak - win)) & (z_base < (z_peak + win))]
        z_med = float(np.nanmedian(sel)) if sel.size >= 150 else float(np.nanmedian(z_base))
        return z_med, (u0, v0, u1, v1)

    @staticmethod
    def _yaw_from_min_area_rect_long(xy: np.ndarray, aspect_min: float, min_pts: int):
        """
        Ritorna yaw_long (asse lungo) usando minAreaRect.
        aspect = long/short.
        """
        if xy.shape[0] < min_pts:
            return None, None
        pts = xy.astype(np.float32).reshape(-1, 1, 2)
        try:
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)  # 4x2
        except Exception:
            return None, None

        v01 = box[1] - box[0]
        v12 = box[2] - box[1]
        l01 = float(np.linalg.norm(v01))
        l12 = float(np.linalg.norm(v12))
        if l01 < 1e-6 or l12 < 1e-6:
            return None, None

        if l01 >= l12:
            v_long = v01
            long_len = l01
            short_len = l12
        else:
            v_long = v12
            long_len = l12
            short_len = l01

        aspect = float(long_len / max(short_len, 1e-6))
        if aspect < aspect_min:
            return None, aspect

        yaw_long = wrap_pi(math.atan2(float(v_long[1]), float(v_long[0])))
        return yaw_long, aspect

    @staticmethod
    def _yaw_from_pca_major(xy: np.ndarray, anis_min: float, min_pts: int):
        """
        PCA: ritorna yaw_major (asse maggiore = lungo).
        anis = lambda_max/lambda_min
        """
        if xy.shape[0] < min_pts:
            return None, None
        mu = np.mean(xy, axis=0)
        X = xy - mu
        C = (X.T @ X) / max(1, X.shape[0] - 1)
        w, v = np.linalg.eigh(C)  # w[0] <= w[1]
        anis = float((w[1] + 1e-12) / (w[0] + 1e-12))
        if anis < anis_min:
            return None, anis
        v_major = v[:, 1]
        yaw = wrap_pi(math.atan2(float(v_major[1]), float(v_major[0])))
        return yaw, anis

    @staticmethod
    def _densest_xy_center(x: np.ndarray, y: np.ndarray, bin_size: float = 0.008):
        if x.size < 120:
            return float(np.nanmedian(x)), float(np.nanmedian(y))
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]; y = y[m]
        if x.size < 120:
            return float(np.nanmedian(x)), float(np.nanmedian(y))

        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))

        if (xmax - xmin) < 1e-4 or (ymax - ymin) < 1e-4:
            return float(np.mean(x)), float(np.mean(y))

        nx = int(max(3, min(140, math.ceil((xmax - xmin) / max(bin_size, 1e-6)))))
        ny = int(max(3, min(140, math.ceil((ymax - ymin) / max(bin_size, 1e-6)))))

        xi = np.clip(((x - xmin) / (xmax - xmin + 1e-12) * (nx - 1)).astype(np.int32), 0, nx - 1)
        yi = np.clip(((y - ymin) / (ymax - ymin + 1e-12) * (ny - 1)).astype(np.int32), 0, ny - 1)

        H = np.zeros((nx, ny), dtype=np.int32)
        for k in range(xi.size):
            H[xi[k], yi[k]] += 1

        bx, by = np.unravel_index(int(np.argmax(H)), H.shape)
        cx = xmin + (bx + 0.5) * (xmax - xmin) / nx
        cy = ymin + (by + 0.5) * (ymax - ymin) / ny
        return float(cx), float(cy)

    @staticmethod
    def _refine_center_in_radius(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                 cx: float, cy: float, r: float = 0.040):
        dx = x - cx
        dy = y - cy
        m = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(z) & ((dx * dx + dy * dy) < (r * r))
        if np.count_nonzero(m) < 80:
            return float(np.nanmedian(x)), float(np.nanmedian(y)), float(np.nanmedian(z)), m
        return float(np.nanmedian(x[m])), float(np.nanmedian(y[m])), float(np.nanmedian(z[m])), m

    @staticmethod
    def _make_uid(name: str, pose: PoseStamped) -> str:
        s = f"{name}|{pose.pose.position.x:.4f}|{pose.pose.position.y:.4f}|{pose.pose.position.z:.4f}"
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

    def _publish_once(self, best):
        if (rospy.Time.now() - self._last_pub).to_sec() < self.min_pub_period:
            return False

        self.pub_obj_pose.publish(best["pose"])
        self.pub_obj_name.publish(String(data=best["name"]))

        rpy = Float64MultiArray()
        rpy.data = [
            best["pose"].pose.position.x,
            best["pose"].pose.position.y,
            best["pose"].pose.position.z,
            float(math.pi), 0.0, float(best["yaw"])
        ]
        self.pub_obj_rpy.publish(rpy)

        uid = self._make_uid(best["name"], best["pose"])
        self.pub_obj_uid.publish(String(data=uid))

        self._last_pub = rospy.Time.now()
        return True

    @staticmethod
    def _snap_angle(a: float, step: float) -> float:
        if step is None or step <= 1e-6:
            return wrap_pi(a)
        return wrap_pi(round(a / step) * step)

    def _apply_yaw_mode_and_offset(self, yaw_long: float) -> float:
        yaw = float(yaw_long)

        # long/short
        if self.yaw_mode == "short":
            yaw = wrap_pi(yaw + math.pi / 2.0)

        # tool offset
        yaw = wrap_pi(yaw + self.yaw_tool_offset)

        # snap (prima del filtro)
        if self.yaw_snap_enable:
            yaw = self._snap_angle(yaw, self.yaw_snap_step)

        # smoothing (spazio circolare)
        if self.yaw_smooth:
            if self._yaw_last is None:
                self._yaw_last = yaw
            else:
                a = float(self.yaw_smooth_alpha)
                s = (1.0 - a) * math.sin(self._yaw_last) + a * math.sin(yaw)
                c = (1.0 - a) * math.cos(self._yaw_last) + a * math.cos(yaw)
                self._yaw_last = wrap_pi(math.atan2(s, c))
            yaw = float(self._yaw_last)

            # snap anche dopo smoothing (per mantenere valori puliti)
            if self.yaw_snap_enable:
                yaw = self._snap_angle(yaw, self.yaw_snap_step)

        return float(yaw)

    # ---------------- main callback ----------------
    def _synced_cb(self, color_msg: Image, depth_msg: Image):
        if self.K0 is None:
            rospy.logwarn_throttle(5.0, "[perception6d] waiting for camera_info...")
            return

        # timeout request
        if self.publish_on_request and self._req_pending and rospy.Time.now() > self._req_deadline:
            rospy.logwarn("[perception6d] request timeout: no publish in %.1fs (had %d frames)",
                          self.request_timeout_s, len(self._req_buf))
            self._req_pending = False
            self._req_buf.clear()
            self._yaw_last = None  # reset anche qui

        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        except CvBridgeError:
            return

        debug_img = color.copy()

        try:
            depth = self._depth_to_meters(depth_msg)
        except Exception:
            self._publish_debug(debug_img, depth_msg.header.frame_id or "camera")
            return

        stamp = color_msg.header.stamp if color_msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        cam_frame = depth_msg.header.frame_id or color_msg.header.frame_id or "zed2_left_camera_optical_frame"

        T_base_cam = self._tf_to_base(cam_frame, stamp)
        if T_base_cam is None:
            rospy.logwarn_throttle(1.0, "[perception6d] TF missing: %s <- %s", self.frame_base, cam_frame)
            self._publish_debug(debug_img, cam_frame)
            return

        hc, wc = color.shape[:2]
        hd, wd = depth.shape[:2]
        fx, fy, cx, cy = self._intrinsics_for_image(wd, hd)

        # table z
        z_table_est, table_roi_px = self._estimate_table_z_base(depth, T_base_cam, fx, fy, cx, cy)

        # draw table ROI
        if self.debug_enable and z_table_est is not None and np.isfinite(z_table_est):
            u0d, v0d, u1d, v1d = table_roi_px
            sx = float(wc) / float(wd)
            sy = float(hc) / float(hd)
            u0c = int(round(u0d * sx)); u1c = int(round(u1d * sx))
            v0c = int(round(v0d * sy)); v1c = int(round(v1d * sy))
            cv2.rectangle(debug_img, (u0c, v0c), (u1c, v1c), (255, 0, 0), 2)
            cv2.putText(debug_img, f"z_table={z_table_est:.3f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # YOLO
        try:
            results = self.model.predict(source=color, conf=self.conf_thresh, imgsz=640, verbose=False)
            if not results:
                self._publish_debug(debug_img, cam_frame)
                self._yaw_last = None
                return
            res = results[0]
        except Exception:
            self._publish_debug(debug_img, cam_frame)
            self._yaw_last = None
            return

        boxes_xyxy = getattr(res.boxes, "xyxy", None)
        if boxes_xyxy is None:
            self._publish_debug(debug_img, cam_frame)
            self._yaw_last = None
            return
        boxes_xyxy = np.array(boxes_xyxy)

        sx_cd = float(wd) / float(wc)
        sy_cd = float(hd) / float(hc)

        # robot ROI in color
        if self.robot_roi_enable:
            rux0 = int(self.robot_roi_u0 * wc)
            rux1 = int(self.robot_roi_u1 * wc)
            rvy0 = int(self.robot_roi_v0 * hc)
            rvy1 = int(self.robot_roi_v1 * hc)
            cv2.rectangle(debug_img, (rux0, rvy0), (rux1, rvy1), (0, 0, 255), 2)
        else:
            rux0 = rux1 = rvy0 = rvy1 = -1

        rej_no_depth = rej_robot_roi = rej_table_z = rej_table_xy = 0
        candidates = []

        for i, b in enumerate(boxes_xyxy):
            try:
                x1c = int(b[0]); y1c = int(b[1]); x2c = int(b[2]); y2c = int(b[3])
            except Exception:
                continue

            x1c = max(0, min(wc - 1, x1c))
            x2c = max(0, min(wc - 1, x2c))
            y1c = max(0, min(hc - 1, y1c))
            y2c = max(0, min(hc - 1, y2c))
            if x2c <= x1c + 4 or y2c <= y1c + 4:
                continue

            # class & conf
            try:
                class_id = int(res.boxes.cls[i].item()) if hasattr(res.boxes.cls[i], "item") else int(res.boxes.cls[i])
                class_name = res.names[class_id] if hasattr(res, "names") and class_id in res.names else str(class_id)
            except Exception:
                class_name = "obj"

            try:
                conf = float(res.boxes.conf[i].item()) if hasattr(res.boxes.conf[i], "item") else float(res.boxes.conf[i])
            except Exception:
                conf = 0.0

            # disegna sempre bbox+label
            self._draw_label(debug_img, x1c, y1c, x2c, y2c, f"{class_name} {conf:.2f}", (0, 255, 0), 2)

            # robot ROI reject
            if self.robot_roi_enable:
                cx_box = int(0.5 * (x1c + x2c))
                cy_box = int(0.5 * (y1c + y2c))
                if (rux0 <= cx_box <= rux1) and (rvy0 <= cy_box <= rvy1):
                    rej_robot_roi += 1
                    self._draw_label(debug_img, x1c, y1c, x2c, y2c, f"{class_name} REJ robotROI", (0, 0, 255), 2)
                    continue

            # bbox in depth
            x1d = int(round(x1c * sx_cd)); x2d = int(round(x2c * sx_cd))
            y1d = int(round(y1c * sy_cd)); y2d = int(round(y2c * sy_cd))
            x1d = max(0, min(wd - 1, x1d)); x2d = max(0, min(wd - 1, x2d))
            y1d = max(0, min(hd - 1, y1d)); y2d = max(0, min(hd - 1, y2d))
            if x2d <= x1d + 4 or y2d <= y1d + 4:
                continue

            # shrink bbox
            bw = (x2d - x1d)
            bh = (y2d - y1d)
            shx = int(round(self.bbox_shrink * bw))
            shy = int(round(self.bbox_shrink * bh))
            x1s = max(0, min(wd - 1, x1d + shx))
            x2s = max(0, min(wd - 1, x2d - shx))
            y1s = max(0, min(hd - 1, y1d + shy))
            y2s = max(0, min(hd - 1, y2d - shy))
            if x2s <= x1s + 4 or y2s <= y1s + 4:
                x1s, x2s, y1s, y2s = x1d, x2d, y1d, y2d

            samp = self._sample_bbox_points(depth, x1s, y1s, x2s, y2s, max_pts=3200)
            if samp is None:
                rej_no_depth += 1
                self._draw_label(debug_img, x1c, y1c, x2c, y2c, f"{class_name} REJ noDepth", (0, 0, 255), 2)
                continue

            us, vs, zs = samp

            # camera->base
            X = (us - cx) * zs / fx
            Y = (vs - cy) * zs / fy
            Z = zs
            ones = np.ones_like(Z, dtype=np.float32)
            pts_cam = np.stack([X, Y, Z, ones], axis=0)
            pts_base = T_base_cam @ pts_cam

            xb = pts_base[0, :]
            yb = pts_base[1, :]
            zb = pts_base[2, :]

            # default
            x_med = float(np.nanmedian(xb))
            y_med = float(np.nanmedian(yb))
            z_med = float(np.nanmedian(zb))
            yaw_out = 0.0

            have_obj_pts = False
            yaw_dbg = ""

            # selezione oggetto sopra tavolo + top surface
            if self.use_table_z and (z_table_est is not None) and np.isfinite(z_table_est) and self.use_obj_range_filter:
                zt = float(z_table_est)
                dz = zb - zt

                obj = np.isfinite(dz) & (dz > self.table_obj_gap) & (dz < self.table_obj_max)
                if np.count_nonzero(obj) >= 180:
                    have_obj_pts = True

                    # TOP usato per XY (precisione)
                    dz_obj = dz[obj]
                    thr = float(np.percentile(dz_obj, np.clip(self.top_keep_percentile, 40.0, 95.0)))
                    top = obj & (dz >= thr)
                    use_xy = top if np.count_nonzero(top) >= 140 else obj

                    xu = xb[use_xy]
                    yu = yb[use_xy]
                    zu = zb[use_xy]

                    # 1) densest-bin center
                    cx_d, cy_d = self._densest_xy_center(xu, yu, bin_size=self.dense_bin_size)
                    # 2) refine within radius (median)
                    x_med, y_med, z_med, _ = self._refine_center_in_radius(xu, yu, zu, cx_d, cy_d, r=self.xy_inlier_r)

                    # YAW robusto: usa footprint completa (obj)
                    if self.estimate_yaw:
                        if self.yaw_use_obj_points:
                            xy_yaw = np.stack([xb[obj], yb[obj]], axis=1)
                        else:
                            xy_yaw = np.stack([xu, yu], axis=1)

                        yaw_long = None

                        if self.yaw_use_rect:
                            yr, asp = self._yaw_from_min_area_rect_long(
                                xy_yaw, self.yaw_rect_min_aspect, self.yaw_rect_min_pts
                            )
                            if yr is not None:
                                yaw_long = float(yr)
                                yaw_dbg = f"rect asp={asp:.2f}"

                        if yaw_long is None and self.yaw_use_pca:
                            yp, anis = self._yaw_from_pca_major(
                                xy_yaw, self.yaw_min_anisotropy, self.yaw_pca_min_pts
                            )
                            if yp is not None:
                                yaw_long = float(yp)
                                yaw_dbg = f"pca an={anis:.2f}"
                            else:
                                yaw_dbg = f"fail an={anis:.2f}" if anis is not None else "fail"

                        if yaw_long is not None:
                            yaw_out = self._apply_yaw_mode_and_offset(yaw_long)

            # filtro Z
            if self.use_table_z and (z_table_est is not None) and np.isfinite(z_table_est):
                zt = float(z_table_est)
                if have_obj_pts:
                    dzm = z_med - zt
                    if not (dzm > self.table_obj_gap and dzm < self.table_obj_max):
                        rej_table_z += 1
                        self._draw_label(debug_img, x1c, y1c, x2c, y2c, f"{class_name} REJ Zrng", (0, 180, 255), 2)
                        continue
                else:
                    if abs(z_med - zt) > self.table_z_tol:
                        rej_table_z += 1
                        self._draw_label(debug_img, x1c, y1c, x2c, y2c, f"{class_name} REJ Z", (0, 180, 255), 2)
                        continue

            # filtro XY bounds
            if self.use_table_xy:
                if not (self.table_x_min <= x_med <= self.table_x_max and self.table_y_min <= y_med <= self.table_y_max):
                    rej_table_xy += 1
                    self._draw_label(debug_img, x1c, y1c, x2c, y2c, f"{class_name} REJ XY", (0, 180, 255), 2)
                    continue

            # bias
            x_out = x_med + self.grasp_bias_x
            y_out = y_med + self.grasp_bias_y
            z_out = z_med

            # score: preferisci vicino al tavolo + conf alta
            z_term = 0.0 if (z_table_est is None or not np.isfinite(z_table_est)) else abs(z_out - float(z_table_est))
            score = float(z_term) + 0.25 * (1.0 - float(conf))

            # yaw debug text
            if self.estimate_yaw and yaw_dbg:
                self._draw_label(
                    debug_img,
                    x1c, min(hc - 1, y2c + 2), x2c, min(hc - 1, y2c + 22),
                    f"yaw={yaw_out:+.2f} {yaw_dbg} mode={self.yaw_mode}",
                    (0, 255, 255), 2
                )

            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.frame_base
            pose.pose.position.x = x_out
            pose.pose.position.y = y_out
            pose.pose.position.z = z_out
            q = tf.transformations.quaternion_from_euler(math.pi, 0.0, float(yaw_out))
            pose.pose.orientation.x = float(q[0])
            pose.pose.orientation.y = float(q[1])
            pose.pose.orientation.z = float(q[2])
            pose.pose.orientation.w = float(q[3])

            candidates.append({
                "name": class_name,
                "conf": float(conf),
                "yaw": float(yaw_out),
                "pose": pose,
                "score": float(score),
                "z_table": None if z_table_est is None else float(z_table_est),
            })

        # publish debug sempre (con bbox/label)
        self._publish_debug(debug_img, cam_frame)

        # pick best
        if candidates:
            candidates.sort(key=lambda d: d["score"])
            self._best = candidates[0]
            self._have_best = True
        else:
            self._best = None
            self._have_best = False
            self._yaw_last = None  # reset filtro se non vede nulla

        # log periodico
        now = rospy.Time.now()
        if (now - self._last_print).to_sec() > self.debug_print_every:
            self._last_print = now
            zt = "None" if (z_table_est is None or not np.isfinite(z_table_est)) else f"{float(z_table_est):.3f}"
            if candidates:
                rospy.loginfo("[perception6d] cand=%d best=%s conf=%.2f yaw=%.2f z_table=%s rejs: noDepth=%d robotROI=%d tableZ=%d tableXY=%d",
                              len(candidates), candidates[0]["name"], candidates[0]["conf"], candidates[0]["yaw"], zt,
                              rej_no_depth, rej_robot_roi, rej_table_z, rej_table_xy)
            else:
                rospy.logwarn("[perception6d] NO CAND. z_table=%s rejs: noDepth=%d robotROI=%d tableZ=%d tableXY=%d",
                              zt, rej_no_depth, rej_robot_roi, rej_table_z, rej_table_xy)

        # publish /vision
        if not self.publish_on_request:
            if self._have_best:
                self._publish_once(self._best)
            return

        # request-mode: accumula frame e pubblica mediana
        if self._req_pending and self._have_best:
            p = self._best["pose"].pose.position
            self._req_buf.append((
                float(p.x), float(p.y), float(p.z),
                float(self._best["yaw"]), float(self._best["conf"]), self._best["name"]
            ))

            elapsed = (rospy.Time.now() - self._req_start_t).to_sec()
            have_min = (len(self._req_buf) >= self.req_min_frames)
            have_full = (len(self._req_buf) >= self.req_collect_N)
            time_to_publish = have_full or (have_min and elapsed >= self.req_max_wait_s)

            if time_to_publish:
                xs = np.array([t[0] for t in self._req_buf], dtype=np.float32)
                ys = np.array([t[1] for t in self._req_buf], dtype=np.float32)
                zs = np.array([t[2] for t in self._req_buf], dtype=np.float32)
                yaws = np.array([t[3] for t in self._req_buf], dtype=np.float32)
                confs = np.array([t[4] for t in self._req_buf], dtype=np.float32)

                x_med = float(np.median(xs))
                y_med = float(np.median(ys))
                z_med = float(np.median(zs))

                # yaw: mediana circolare + outlier reject
                s0 = float(np.median(np.sin(yaws)))
                c0 = float(np.median(np.cos(yaws)))
                yaw0 = wrap_pi(math.atan2(s0, c0))

                thr = math.radians(float(self.yaw_outlier_deg))
                keep = np.array([abs(wrap_pi(float(yy) - yaw0)) < thr for yy in yaws], dtype=bool)

                if np.count_nonzero(keep) >= max(2, int(0.6 * yaws.size)):
                    yk = yaws[keep]
                    s = float(np.median(np.sin(yk)))
                    c = float(np.median(np.cos(yk)))
                    yaw_med = wrap_pi(math.atan2(s, c))
                else:
                    yaw_med = yaw0

                # snap finale (stesso schema del frame singolo)
                if self.yaw_snap_enable:
                    yaw_med = self._snap_angle(yaw_med, self.yaw_snap_step)

                name_out = self._best["name"]
                conf_out = float(np.max(confs))

                pose = self._best["pose"]
                pose.pose.position.x = x_med
                pose.pose.position.y = y_med
                pose.pose.position.z = z_med
                q = tf.transformations.quaternion_from_euler(math.pi, 0.0, yaw_med)
                pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = map(float, q)

                best_pub = dict(self._best)
                best_pub["pose"] = pose
                best_pub["conf"] = conf_out
                best_pub["yaw"] = float(yaw_med)
                best_pub["name"] = name_out

                if self._publish_once(best_pub):
                    zt = "None" if best_pub["z_table"] is None else f"{best_pub['z_table']:.3f}"
                    rospy.loginfo("[perception6d] PUBLISHED request median (%d frames, %.2fs) name=%s conf=%.2f pos=(%.3f,%.3f,%.3f) yaw=%.3f z_table=%s",
                                  len(self._req_buf), elapsed, name_out, conf_out, x_med, y_med, z_med, float(yaw_med), zt)
                    self._req_pending = False
                    self._req_buf.clear()
                    self._yaw_last = None  # reset per prossima richiesta

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        Perception6DNode().spin()
    except rospy.ROSInterruptException:
        pass
    except Exception:
        pass