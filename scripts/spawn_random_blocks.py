#!/usr/bin/env python3

# --------------------------------------------------
# HOW TO START THIS SCRIPT
#cd ~/ros_ws
#source devel/setup.bash
#export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/root/ros_ws/src/lab_myproject/models
#python3 -i src/locosim/robot_control/base_controllers/ur5_generic.py
#rosrun lab_myproject spawn_random_blocks.py
# --------------------------------------------------

import rospy
import os
import random
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

N_BLOCKS = 5

TABLE_Z = 0.82        # height of the table surface
SPAWN_Z = TABLE_Z + 0.05

X_RANGE = (0.45, 0.75)
Y_RANGE = (-0.25, 0.25)

MODELS_DIR = os.path.expanduser(
    "~/ros_ws/src/lab_myproject/models"
)

# --------------------------------------------------
# UTILS
# --------------------------------------------------

def get_available_models():
    models = []
    for name in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, name)
        sdf_file = os.path.join(model_path, "model.sdf")
        if os.path.isdir(model_path) and os.path.isfile(sdf_file):
            models.append(name)
    return models


def random_pose():
    pose = Pose()
    pose.position.x = random.uniform(*X_RANGE)
    pose.position.y = random.uniform(*Y_RANGE)
    pose.position.z = SPAWN_Z
    pose.orientation.w = 1.0
    return pose


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    rospy.init_node("spawn_random_blocks")

    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    spawn_srv = rospy.ServiceProxy(
        "/gazebo/spawn_sdf_model",
        SpawnModel
    )

    available_models = get_available_models()

    if not available_models:
        rospy.logerr("No models found in models directory!")
        exit(1)

    rospy.loginfo(f"Available models: {available_models}")

    for i in range(N_BLOCKS):

        model_name = random.choice(available_models)
        instance_name = f"{model_name}_{i}"

        sdf_path = os.path.join(
            MODELS_DIR,
            model_name,
            "model.sdf"
        )

        with open(sdf_path, "r") as f:
            model_xml = f.read()

        try:
            spawn_srv(
                model_name=instance_name,
                model_xml=model_xml,
                robot_namespace="",
                initial_pose=random_pose(),
                reference_frame="world"
            )
            rospy.loginfo(f"Spawned {instance_name}")

        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to spawn {instance_name}: {e}")
