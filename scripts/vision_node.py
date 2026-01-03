#!/usr/bin/env python3
import rospy
import numpy as np
from collections import deque

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from visualization_msgs.msg import Marker

# -------------------------------
# PARAMETRI
# -------------------------------
CLUSTER_DIST = 0.05        # metri
MIN_CLUSTER_SIZE = 150

X_RANGE = (0.05, 0.70)
Y_RANGE = (0.20, 0.75)
TABLE_Z = 0.82


class VisionNode:

    def __init__(self):

        rospy.loginfo("Vision node started, waiting for point cloud...")

        rospy.init_node("vision_node")

        self.cloud_sub = rospy.Subscriber(
            "/ur5/zed_node/point_cloud/cloud_registered",
            PointCloud2,
            self.cloud_callback,
            queue_size=1
        )

        self.marker_pub = rospy.Publisher(
            "/vision/blocks",
            Marker,
            queue_size=10
        )

        rospy.loginfo("Vision node with clustering started")

    # -------------------------------
    def cloud_callback(self, msg):
        self.cloud_frame = msg.header.frame_id

        points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append(p)

        if len(points) < 1000:
            return

        pts = np.array(points)
        centroid = pts.mean(axis=0)

        self.publish_marker(centroid, 0)

    # -------------------------------
    def euclidean_clustering(self, pts):
        clusters = []
        visited = np.zeros(len(pts), dtype=bool)

        for i in range(len(pts)):
            if visited[i]:
                continue

            queue = deque([i])
            visited[i] = True
            cluster = []

            while queue:
                idx = queue.popleft()
                cluster.append(pts[idx])

                dists = np.linalg.norm(pts - pts[idx], axis=1)
                neighbors = np.where((dists < CLUSTER_DIST) & (~visited))[0]

                for n in neighbors:
                    visited[n] = True
                    queue.append(n)

            if len(cluster) > MIN_CLUSTER_SIZE:
                clusters.append(np.array(cluster))

        return clusters

    # -------------------------------
    def publish_clusters(self, clusters):
        for i, cluster in enumerate(clusters):
            centroid = cluster.mean(axis=0)
            self.publish_marker(centroid, i)

    # -------------------------------
    def publish_marker(self, pos, mid):
        m = Marker()
        m.header.frame_id = self.cloud_frame
        m.header.stamp = rospy.Time.now()

        m.ns = "blocks"
        m.id = mid

        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.pose.position.x = pos[0]
        m.pose.position.y = pos[1]
        m.pose.position.z = pos[2]

        m.pose.orientation.w = 1.0

        m.scale.x = 0.04
        m.scale.y = 0.04
        m.scale.z = 0.04

        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0

        self.marker_pub.publish(m)


if __name__ == "__main__":
    VisionNode()
    rospy.spin()
