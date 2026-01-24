#!/usr/bin/env python3
import rospy
import tf
from gazebo_msgs.srv import GetModelState

def main():
    rospy.init_node("gazebo_world_to_base_tf")

    model_name = rospy.get_param("~model_name", "ur5")
    base_frame = rospy.get_param("~base_frame", "base_link_inertia")
    world_frame = rospy.get_param("~world_frame", "world")

    rospy.wait_for_service("/gazebo/get_model_state")
    get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    br = tf.TransformBroadcaster()

    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        try:
            resp = get_state(model_name, world_frame)
            p = resp.pose.position
            q = resp.pose.orientation
            br.sendTransform((p.x, p.y, p.z),
                             (q.x, q.y, q.z, q.w),
                             rospy.Time.now(),
                             base_frame,
                             world_frame)
        except Exception as e:
            rospy.logwarn_throttle(5, f"get_model_state failed: {e}")
        rate.sleep()

if __name__ == "__main__":
    main()
