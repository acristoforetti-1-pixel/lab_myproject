#!/usr/bin/env python3

# --------------------------------------------------
# HOW TO START THIS SCRIPT
#cd ~/ros_ws
#source devel/setup.bash
#export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/root/ros_ws/src/lab_myproject/models
#python3 -i src/locosim/robot_control/base_controllers/ur5_generic.py
#rosrun lab_myproject spawn_random_blocks.py
# --------------------------------------------------
#!/usr/bin/env python3
import rospy
import os
import random
import math
import uuid

from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose

import tf.transformations as tft

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

N_BLOCKS = 5
MIN_DIST = 0.07          # 7 cm tra i centri
MARGIN = 0.02            # margine dai bordi/parete (6 cm)

TABLE_Z = 0.82
SPAWN_Z = TABLE_Z + 0.03

# restringi un po' l'area utile e poi applichiamo anche MARGIN
X_RANGE = (0.05, 0.70)
Y_RANGE = (0.20, 0.75)

MODELS_DIR = os.path.expanduser("~/ros_ws/src/lab_myproject/models")

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


def sample_xy():
    """Campiona x,y rispettando un margine dai limiti."""
    x = random.uniform(X_RANGE[0] + MARGIN, X_RANGE[1] - MARGIN)
    y = random.uniform(Y_RANGE[0] + MARGIN, Y_RANGE[1] - MARGIN)
    return x, y


def random_pose_non_overlapping(existing_xy, max_tries=500):
    """Cerca una posa che non sia troppo vicina alle altre."""
    for _ in range(max_tries):
        x, y = sample_xy()
        ok = all((x-ex)**2 + (y-ey)**2 >= MIN_DIST**2 for ex, ey in existing_xy)
        if ok:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = SPAWN_Z

            # yaw random per evitare incastri identici
            yaw = random.uniform(-math.pi, math.pi)
            q = tft.quaternion_from_euler(0, 0, yaw)
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            return pose, (x, y)

    # fallback: se non trova spazio, usa comunque un punto (meglio che crashare)
    x, y = sample_xy()
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = SPAWN_Z
    pose.orientation.w = 1.0
    return pose, (x, y)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    rospy.init_node("spawn_random_blocks")

    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    spawn_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

    available_models = get_available_models()
    if not available_models:
        rospy.logerr("No models found in models directory!")
        raise SystemExit(1)

    rospy.loginfo(f"Available models: {available_models}")

    spawned_xy = []

    for i in range(N_BLOCKS):
        model_name = random.choice(available_models)
        instance_name = f"{model_name}_{uuid.uuid4().hex[:8]}"

        sdf_path = os.path.join(MODELS_DIR, model_name, "model.sdf")
        with open(sdf_path, "r") as f:
            model_xml = f.read()

        pose, xy = random_pose_non_overlapping(spawned_xy)
        spawned_xy.append(xy)

        try:
            spawn_srv(
                model_name=instance_name,
                model_xml=model_xml,
                robot_namespace="",
                initial_pose=pose,
                reference_frame="world"
            )
            rospy.loginfo(f"Spawned {instance_name} at x={xy[0]:.3f}, y={xy[1]:.3f}")

        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to spawn {instance_name}: {e}")