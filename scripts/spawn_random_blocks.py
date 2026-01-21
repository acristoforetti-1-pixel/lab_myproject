#!/usr/bin/env python3
import rospy
import os
import random
import math
import uuid

from std_msgs.msg import String
from gazebo_msgs.srv import SpawnModel, GetModelState
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties
from gazebo_msgs.msg import ODEPhysics
from geometry_msgs.msg import Pose, Vector3
from std_msgs.msg import Float64MultiArray
import tf.transformations as tft

ROBOT_MODEL_NAME = "ur5"

N_BLOCKS = 2
MIN_DIST = 0.07
MARGIN = 0.02

TABLE_Z = 0.87
SPAWN_Z = TABLE_Z +0.003
Z_OFFSET = 1.72

# area utile in BASE_LINK (tavolo)
X_RANGE = (-0.35, 0.30)
Y_RANGE = (0.10, 0.40)

#zona raggiungibile (in base_link)
MIN_RXY = 0.22   # sotto questo = troppo vicino alla base (fa casino)
MAX_RXY = 0.55   # sopra questo = spesso jump/singolarita' (se vuoi aumenta)

#  “NO GO ZONE” (zona che di solito crea singularity / fold)
# (regola questi numeri se vuoi, ma già così migliora tanto)
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
    return [
        name for name in os.listdir(MODELS_DIR)
        if os.path.isfile(os.path.join(MODELS_DIR, name, "model.sdf"))
    ]


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
    """Campiona in base_link rispettando range tavolo + rxy + no-go."""
    for _ in range(2000):
        x = random.uniform(X_RANGE[0] + MARGIN, X_RANGE[1] - MARGIN)
        y = random.uniform(Y_RANGE[0] + MARGIN, Y_RANGE[1] - MARGIN)

        rxy = math.sqrt(x*x + y*y)
        if rxy < MIN_RXY or rxy > MAX_RXY:
            continue

        if in_no_go_zone(x, y):
            continue

        return x, y

    # fallback: se non trova niente, torna comunque un punto valido "medio"
    return 0.20, 0.25


def base_to_world(get_state_srv, x_b, y_b, z_b, yaw_b):
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
    z_b_spawn = SPAWN_Z - Z_OFFSET

    for _ in range(500):
        x_b, y_b = sample_xy_base()

        if all((x_b-ex)**2 + (y_b-ey)**2 >= MIN_DIST**2 for ex, ey in existing_xy_base):
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
    rospy.wait_for_service("/gazebo/get_physics_properties")
    rospy.wait_for_service("/gazebo/set_physics_properties")

    getp = rospy.ServiceProxy("/gazebo/get_physics_properties", GetPhysicsProperties)
    setp = rospy.ServiceProxy("/gazebo/set_physics_properties", SetPhysicsProperties)

    cur = getp()

    ode = ODEPhysics()

    #  evita “ancoraggi” / corpi che restano congelati
    ode.auto_disable_bodies = False

    #  solver più robusto (contatti più stabili nelle pinze)
    ode.sor_pgs_iters = 200
    ode.sor_pgs_w = 1.2
    ode.sor_pgs_rms_error_tol = 0.0

    # contatto più “morbido”
    ode.erp = 0.15        # più basso = meno rimbalzo/rigidità
    ode.cfm = 1e-5        # piccolo >0 = contatti meno “cemento”

    # riduce incastri e compenetrazioni
    ode.contact_surface_layer = 0.002
    ode.contact_max_correcting_vel = 0.6

    ode.max_contacts = 80

    #  timestep più fine = meno tunneling
    time_step = 0.0005        # 0.5 ms
    max_update_rate = 2000.0  # 2000 Hz

    gravity = cur.gravity if cur.gravity else Vector3(0, 0, -9.81)

    ok = setp(time_step, max_update_rate, gravity, ode)
    if ok.success:
        rospy.loginfo("Gazebo physics forced: dt=%.4f rate=%.0f iters=%d ERP=%.3f CFM=%.1e",
                      time_step, max_update_rate, ode.sor_pgs_iters, ode.erp, ode.cfm)
    else:
        rospy.logwarn("⚠️ Physics not changed: %s", ok.status_message)
# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    rospy.init_node("spawn_random_blocks")

    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    rospy.wait_for_service("/gazebo/get_model_state")

    spawn_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    get_state_srv = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

    pub_obj_rpy = rospy.Publisher("/vision/object_rpy", Float64MultiArray, queue_size=10)
    pub_obj_name = rospy.Publisher("/vision/object_name", String, queue_size=10)
    # ✅ forza fisica ogni volta che spawni
    fix_gazebo_physics()

    models = get_available_models()
    if not models:
        rospy.logerr("No models found!")
        exit(1)

    rospy.loginfo(f"Available models: {models}")
    rospy.loginfo(f"Spawn area base_link: X{X_RANGE} Y{Y_RANGE} rxy[{MIN_RXY},{MAX_RXY}]")
    rospy.loginfo(f"NO-GO rect: {NO_GO_RECT}")

    spawned_xy_base = []

    for _ in range(N_BLOCKS):
        model = random.choice(models)
        instance = f"{model}_{uuid.uuid4().hex[:8]}"

        with open(os.path.join(MODELS_DIR, model, "model.sdf")) as f:
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

        q = pose_w.orientation
        yaw_w = yaw_from_quat(q)

        x_b, y_b, z_b, yaw_b = world_to_base(
            get_state_srv,
            pose_w.position.x,
            pose_w.position.y,
            pose_w.position.z,
            yaw_w
        )

        msg = Float64MultiArray()
        msg.data = [x_b, y_b, z_b, math.pi, 0.0, yaw_b]

        # ✅ PUBBLICA PER 1s così il task node lo riceve sicuro
        # ✅ aspetta che il blocco si assesti sul tavolo
        rospy.sleep(0.4)

        # ✅ PUBBLICA PER 1 SECONDO
        name_msg = String()
        name_msg.data = model  # es: "X1-Y2-Z2-FILLET"
        rate = rospy.Rate(10)
        t0 = rospy.Time.now()

        while (rospy.Time.now() - t0).to_sec() < 1.0:
            pub_obj_rpy.publish(msg)
            pub_obj_name.publish(name_msg)
            rate.sleep()

        rospy.loginfo(f"Spawned: {instance} base_link x={x_b:.3f} y={y_b:.3f} rxy={math.sqrt(x_b*x_b+y_b*y_b):.3f} yaw={yaw_b:.3f}")