#!/usr/bin/env python3

import rospy
import numpy as np

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# ---------------- CONFIG ----------------
TABLE_Z = 0.82
Z_TOL = 0.03   # 3 cm sopra il tavolo

# ---------------- GLOBALS ----------------
pub_marker = None


def cloud_callback(msg):
    rospy.loginfo_throttle(2.0, "PointCloud received")

    points = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        x, y, z = p
        if abs(z - TABLE_Z) < Z_TOL:
            points.append([x, y, z])

    rospy.loginfo_throttle(2.0, f"Filtered points: {len(points)}")

    if len(points) == 0:
        return

    pts = np.array(points)
    centroid = np.mean(pts, axis=0)

    publish_marker(centroid)


def publish_marker(pos):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "vision"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    marker.pose.position.x = pos[0]
    marker.pose.position.y = pos[1]
    marker.pose.position.z = pos[2]
    marker.pose.orientation.w = 1.0

    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05

    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    marker.color.a = 1.0

    pub_marker.publish(marker)


def main():
    global pub_marker

    rospy.init_node("vision_node")
    rospy.loginfo("Vision node started")

    rospy.Subscriber(
        "/ur5/zed_node/point_cloud/cloud_registered",
        PointCloud2,
        cloud_callback,
        queue_size=1
    )

    pub_marker = rospy.Publisher(
        "/vision/centroid",
        Marker,
        queue_size=1
    )

    rospy.spin()


if __name__ == "__main__":
    main()
