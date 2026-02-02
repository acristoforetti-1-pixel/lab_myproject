#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file spawn_random_blocks.py
@brief Random spawning of rigid blocks in Gazebo for pick-and-place experiments.

This script populates the Gazebo simulation environment with rigid,
non-deformable blocks randomly placed on a table surface. Spawned objects
are guaranteed to lie within the robot reachable workspace, avoid predefined
no-go regions, and not overlap with each other.

Object poses are sampled in the robot base frame and converted to the Gazebo
world frame using the robot model state. The script is intended for automated
testing and benchmarking of perception, planning, and manipulation pipelines
in a pick-and-place scenario.

Main features:
- Random sampling in the base_link frame
- Reachability and exclusion zone constraints
- Non-overlapping object placement
- On-the-fly adjustment of Gazebo physics parameters
- Support for multiple block models loaded from disk

This script does not publish perception outputs and is intended for offline
environment setup only.
"""

import rospy
import os
import random
import math
import uuid

from gazebo_msgs.srv import SpawnModel, GetModelState
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties
from gazebo_msgs.msg import ODEPhysics
from geometry_msgs.msg import Pose, Vector3
import tf.transformations as tft

ROBOT_MODEL_NAME = "ur5"

# ---- spawn settings ----
N_BLOCKS = 1
MIN_DIST = 0.07
MARGIN = 0.02

TABLE_Z = 0.87
SPAWN_Z = TABLE_Z + 0.003
Z_OFFSET = 1.72

# area utile in BASE_LINK (tavolo)
X_RANGE = (-0.35, 0.30)
Y_RANGE = (0.10, 0.40)

# zona raggiungibile (in base_link)
MIN_RXY = 0.22
MAX_RXY = 0.55

NO_GO_RECT = {
    "x_min": -0.35,
    "x_max": -0.18,
    "y_min":  0.10,
    "y_max":  0.22
}

MODELS_DIR = os.path.expanduser("~/ros_ws/src/lab_myproject/models")


# -------------------------------------------------------
# UTILS
# -------------------------------------------------------
def get_available_models():
    """
    @brief Retrieve all available block models.

    Scans the models directory and returns the names of all Gazebo models
    that provide a valid SDF description. Each model corresponds to a rigid
    block that can be spawned in the simulation.

    @return List of available model names.
    """
    if not os.path.isdir(MODELS_DIR):
        return []
    out = []
    for name in os.listdir(MODELS_DIR):
        sdf = os.path.join(MODELS_DIR, name, "model.sdf")
        if os.path.isfile(sdf):
            out.append(name)
    return out


def yaw_from_quat(q):
    return tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]


def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def in_no_go_zone(x, y):
    return (NO_GO_RECT["x_min"] <= x <= NO_GO_RECT["x_max"] and
            NO_GO_RECT["y_min"] <= y <= NO_GO_RECT["y_max"])


def sample_xy_base():
    """
    @brief Sample a valid (x, y) position in the robot base frame.

    The sampling respects table boundaries, radial reachability constraints,
    and predefined no-go regions. The function retries multiple times before
    returning a fallback position.

    @return A valid (x, y) position expressed in the base_link frame.
    """
    for _ in range(2000):
        x = random.uniform(X_RANGE[0] + MARGIN, X_RANGE[1] - MARGIN)
        y = random.uniform(Y_RANGE[0] + MARGIN, Y_RANGE[1] - MARGIN)

        rxy = math.sqrt(x*x + y*y)
        if rxy < MIN_RXY or rxy > MAX_RXY:
            continue

        if in_no_go_zone(x, y):
            continue

        return x, y

    return 0.20, 0.25


def base_to_world(get_state_srv, x_b, y_b, z_b, yaw_b):
    """
    @brief Convert a pose from base_link frame to Gazebo world frame.

    The conversion uses the current robot pose obtained from Gazebo to map
    coordinates expressed in the robot base frame into the global world frame.
    This ensures consistency between perception, planning, and simulation.

    @param get_state_srv Gazebo service proxy for retrieving robot state.
    @param x_b X position in base_link frame.
    @param y_b Y position in base_link frame.
    @param z_b Z position in base_link frame.
    @param yaw_b Yaw angle in base_link frame.
    @return Pose expressed in the Gazebo world frame.
    """
    st = get_state_srv(model_name=ROBOT_MODEL_NAME, relative_entity_name="world")

    xr = st.pose.position.x
    yr = st.pose.position.y
    yawr = yaw_from_quat(st.pose.orientation)

    c = math.cos(yawr)
    s = math.sin(yawr)

    x_w = xr + c * x_b - s * y_b
    y_w = yr + s * x_b + c * y_b
    z_w = z_b + Z_OFFSET
    yaw_w = wrap_pi(yawr + yaw_b)

    pose = Pose()
    pose.position.x = x_w
    pose.position.y = y_w
    pose.position.z = z_w

    q = tft.quaternion_from_euler(0, 0, yaw_w)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
    return pose


def world_to_base(get_state_srv, x_w, y_w, z_w, yaw_w):
    """Converte world -> base_link (per log/debug)."""
    st = get_state_srv(model_name=ROBOT_MODEL_NAME, relative_entity_name="world")

    xr = st.pose.position.x
    yr = st.pose.position.y
    yawr = yaw_from_quat(st.pose.orientation)

    dx = x_w - xr
    dy = y_w - yr

    c = math.cos(yawr)
    s = math.sin(yawr)

    x_b =  c * dx + s * dy
    y_b = -s * dx + c * dy
    z_b = z_w - Z_OFFSET
    yaw_b = wrap_pi(yaw_w - yawr)

    return x_b, y_b, z_b, yaw_b


def random_pose_non_overlapping(existing_xy_base, get_state_srv):
    """
    @brief Generate a random non-overlapping object pose.

    Samples a valid object pose in the base_link frame while ensuring a
    minimum distance from previously spawned objects. The pose is then
    converted to the Gazebo world frame for spawning.

    @param existing_xy_base List of already occupied (x, y) positions.
    @param get_state_srv Gazebo service proxy.
    @return Tuple containing the world-frame pose and base-frame (x, y).
    """
    z_b_spawn = SPAWN_Z - Z_OFFSET

    for _ in range(500):
        x_b, y_b = sample_xy_base()

        ok = True
        for ex, ey in existing_xy_base:
            if (x_b - ex) ** 2 + (y_b - ey) ** 2 < MIN_DIST ** 2:
                ok = False
                break
        if not ok:
            continue

        yaw_b = random.uniform(-math.pi, math.pi)
        pose_w = base_to_world(get_state_srv, x_b, y_b, z_b_spawn, yaw_b)
        return pose_w, (x_b, y_b)

    # fallback
    x_b, y_b = sample_xy_base()
    pose_w = base_to_world(get_state_srv, x_b, y_b, z_b_spawn, 0.0)
    return pose_w, (x_b, y_b)


# -------------------------------------------------------
# FIX FISICA GAZEBO (senza modificare i modelli)
# -------------------------------------------------------
def fix_gazebo_physics():
    """
    @brief Override Gazebo physics parameters for stable object spawning.

    This function modifies the default Gazebo ODE physics settings to
    improve contact stability and reduce numerical artifacts when spawning
    small rigid objects on planar surfaces.

    The modification avoids the need to edit individual model SDF files.
    """
    rospy.wait_for_service("/gazebo/get_physics_properties")
    rospy.wait_for_service("/gazebo/set_physics_properties")

    getp = rospy.ServiceProxy("/gazebo/get_physics_properties", GetPhysicsProperties)
    setp = rospy.ServiceProxy("/gazebo/set_physics_properties", SetPhysicsProperties)

    cur = getp()

    ode = ODEPhysics()
    ode.auto_disable_bodies = False
    ode.sor_pgs_iters = 200
    ode.sor_pgs_w = 1.2
    ode.sor_pgs_rms_error_tol = 0.0
    ode.erp = 0.15
    ode.cfm = 1e-5
    ode.contact_surface_layer = 0.002
    ode.contact_max_correcting_vel = 0.6
    ode.max_contacts = 80

    time_step = 0.0005
    max_update_rate = 2000.0

    gravity = cur.gravity if cur.gravity else Vector3(0, 0, -9.81)

    ok = setp(time_step, max_update_rate, gravity, ode)
    if ok.success:
        rospy.loginfo("Gazebo physics forced: dt=%.4f rate=%.0f iters=%d ERP=%.3f CFM=%.1e",
                      time_step, max_update_rate, ode.sor_pgs_iters, ode.erp, ode.cfm)
    else:
        rospy.logwarn("Physics not changed: %s", ok.status_message)


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    """
    @brief Main entry point for random block spawning.

    Initializes ROS, configures Gazebo physics, selects random block models,
    and spawns them at valid poses in the simulation environment.
    """
    rospy.init_node("spawn_random_blocks")

    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    rospy.wait_for_service("/gazebo/get_model_state")

    spawn_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    get_state_srv = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

    fix_gazebo_physics()

    models = get_available_models()
    if not models:
        rospy.logerr("No models found in %s", MODELS_DIR)
        return

    rospy.loginfo("Available models: %s", models)
    rospy.loginfo("Spawn area base_link: X%s Y%s rxy[%.2f,%.2f]", X_RANGE, Y_RANGE, MIN_RXY, MAX_RXY)
    rospy.loginfo("NO-GO rect: %s", NO_GO_RECT)

    spawned_xy_base = []

    for _ in range(N_BLOCKS):
        model = random.choice(models)
        instance = f"{model}_{uuid.uuid4().hex[:8]}"

        sdf_path = os.path.join(MODELS_DIR, model, "model.sdf")
        with open(sdf_path, "r") as f:
            model_xml = f.read()

        pose_w, xy_b = random_pose_non_overlapping(spawned_xy_base, get_state_srv)
        spawned_xy_base.append(xy_b)

        spawn_srv(
            model_name=instance,
            model_xml=model_xml,
            robot_namespace="",
            initial_pose=pose_w,
            reference_frame="world"
        )

        # log in base_link (solo debug, NON pubblichiamo /vision/*)
        yaw_w = yaw_from_quat(pose_w.orientation)
        x_b, y_b, z_b, yaw_b = world_to_base(
            get_state_srv,
            pose_w.position.x,
            pose_w.position.y,
            pose_w.position.z,
            yaw_w
        )

        rxy = math.sqrt(x_b*x_b + y_b*y_b)
        rospy.loginfo("Spawned: %s  base_link x=%.3f y=%.3f z=%.3f rxy=%.3f yaw=%.3f",
                      instance, x_b, y_b, z_b, rxy, yaw_b)


if __name__ == "__main__":
    main()
