/*
source ~/ros_ws/devel/setup.bash
rosrun lab_myproject motion_planning_node

source ~/ros_ws/devel/setup.bash
rosrun lab_myproject task_planning_node \
  _object_pose_topic:=/vision/object_pose \
  _gripper_service:=/move_gripper \
  _drop_x:=0.40 _drop_y:=0.00 _drop_z:=0.10 \
  _gripper_open_diameter:=85.0 _gripper_close_diameter:=20.0

# (topic vision dipende da Ale)


source ~/ros_ws/devel/setup.bash
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/root/ros_ws/src/lab_myproject/models
rosrun lab_myproject spawn_random_blocks.py

source ~/ros_ws/devel/setup.bash
rosrun <pacchetto_vision> <nodo_vision>  anche qua dipende da Ale
*/

#include <ros/ros.h>
#include <yaml-cpp/yaml.h>
#include <ros/package.h>

#include <sensor_msgs/JointState.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Bool.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl_parser/kdl_parser.hpp>

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>

#include <ros_impedance_controller/generic_float.h>

#include <map>
#include <mutex>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

class PickPlaceIK {
public:
  PickPlaceIK() : have_js_(false), have_obj_(false), ack_(false) {
    ros::NodeHandle nh, pnh("~");

	// --- Joint limits YAML path
	std::string ur_pkg = ros::package::getPath("ur_description");
	joint_limits_yaml_ = ur_pkg + "/config/ur5e/joint_limits.yaml";


    // --- Topics (match your existing nodes by default)
    pnh.param<std::string>("joint_states_topic", js_topic_, "/ur5/joint_states");
    pnh.param<std::string>("joint_target_topic", target_topic_, "/ur5/joint_target");
    pnh.param<std::string>("ack_topic", ack_topic_, "/acknowledgement");

    // --- Vision inputs (either one works)
    pnh.param<std::string>("object_pose_topic", obj_pose_topic_, "/vision/object_pose"); // PoseStamped
    pnh.param<std::string>("object_rpy_topic",  obj_rpy_topic_,  "/vision/object_rpy");  // Float64MultiArray: x y z r p y

    // --- IK chain
    pnh.param<std::string>("base_link", base_link_, "base_link");
    pnh.param<std::string>("ee_link", ee_link_, "tool0_without_gripper"); // adjust if needed
    pnh.param<std::string>("robot_description_param", robot_desc_param_, "/ur5/robot_description");

    // --- Timing / behavior
    pnh.param("ack_timeout", ack_timeout_, 10.0);

    // --- Approach offsets (meters)
    pnh.param("z_pre",   z_pre_off_,   0.12);
    pnh.param("z_grasp", z_grasp_off_, 0.02);
    pnh.param("z_lift",  z_lift_off_,  0.15);

    // --- Drop location (in base/world frame)
    pnh.param("drop_x", drop_x_, 0.40);
    pnh.param("drop_y", drop_y_, 0.00);
    pnh.param("drop_z", drop_z_, 0.05);

    // --- Orientation policy
    pnh.param("use_only_yaw", use_only_yaw_, true);
    pnh.param("fixed_roll",  fixed_roll_,  M_PI); // often works for top-down depending on EE frame
    pnh.param("fixed_pitch", fixed_pitch_, 0.0);

    // --- Gripper service
    pnh.param<std::string>("gripper_service", gripper_srv_name_, "/move_gripper");
    pnh.param("gripper_open_diameter",  gripper_open_diam_,  70.0); // tune (robotiq max ~85)
    pnh.param("gripper_close_diameter", gripper_close_diam_, 20.0); // tune to your object
    pnh.param("gripper_wait_s", gripper_wait_s_, 0.25);

    // --- ROS I/O
    sub_js_   = nh.subscribe(js_topic_, 1, &PickPlaceIK::jsCb, this);
    sub_ack_  = nh.subscribe(ack_topic_, 1, &PickPlaceIK::ackCb, this);

    sub_obj_pose_ = nh.subscribe(obj_pose_topic_, 1, &PickPlaceIK::objPoseCb, this);
    sub_obj_rpy_  = nh.subscribe(obj_rpy_topic_,  1, &PickPlaceIK::objRpyCb,  this);

    pub_target_ = nh.advertise<std_msgs::Float64MultiArray>(target_topic_, 1);

    // Gripper client
    gripper_cli_ = nh.serviceClient<ros_impedance_controller::generic_float>(gripper_srv_name_);

    // UR5 joint names (update if your joint_states uses different names)
    ur5_names_ = {
      "shoulder_pan_joint",
      "shoulder_lift_joint",
      "elbow_joint",
      "wrist_1_joint",
      "wrist_2_joint",
      "wrist_3_joint"
    };

    // Build IK chain from URDF
    ik_ready_ = initKDL();
    if (!ik_ready_) {
      ROS_ERROR("KDL IK init FAILED. Check robot_description and base/ee link names.");
    } else {
      ROS_INFO("KDL IK ready. base=%s ee=%s", base_link_.c_str(), ee_link_.c_str());
    }

    ROS_INFO("PickPlaceIK ready:");
    ROS_INFO("  vision pose: %s   OR  rpy array: %s", obj_pose_topic_.c_str(), obj_rpy_topic_.c_str());
    ROS_INFO("  joint target pub: %s   ack: %s", target_topic_.c_str(), ack_topic_.c_str());
    ROS_INFO("  gripper service: %s (open=%.1f close=%.1f)", gripper_srv_name_.c_str(),
             gripper_open_diam_, gripper_close_diam_);
  }

  void spin() {
    ros::Rate r(50);
    while (ros::ok()) {
      ros::spinOnce();

      geometry_msgs::PoseStamped obj;
      std::vector<double> q_seed;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!ik_ready_ || !have_js_ || !have_obj_) {
          r.sleep();
          continue;
        }
        obj = obj_pose_base_;
        q_seed = q_cur_;
        have_obj_ = false; // consume once (1 pick per detection)
      }

      doPickPlace(obj, q_seed);
      r.sleep();
    }
  }

private:
  // ------------------ Callbacks ------------------
  void jsCb(const sensor_msgs::JointState& msg) {
    if (msg.name.size() != msg.position.size()) return;

    std::lock_guard<std::mutex> lk(mtx_);

    if (idx_.empty()) {
      idx_.assign(6, -1);
      for (int k = 0; k < 6; ++k) {
        for (int i = 0; i < (int)msg.name.size(); ++i) {
          if (msg.name[i] == ur5_names_[k]) { idx_[k] = i; break; }
        }
      }
      bool ok = true;
      for (int k = 0; k < 6; ++k) ok = ok && (idx_[k] >= 0);
      if (!ok) {
        ROS_ERROR("UR5 joint names not found in joint_states. Update ur5_names_.");
        return;
      }
      ROS_INFO("Joint index map created from /joint_states.");
    }

    q_cur_.resize(6);
    for (int k = 0; k < 6; ++k) q_cur_[k] = msg.position[idx_[k]];
    have_js_ = true;
  }

  void ackCb(const std_msgs::Bool& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    ack_ = msg.data;
  }

  void objPoseCb(const geometry_msgs::PoseStamped& msg) {
    // assume msg is already in base/world as you said
    std::lock_guard<std::mutex> lk(mtx_);
    obj_pose_base_ = msg;
    have_obj_ = true;
  }

  void objRpyCb(const std_msgs::Float64MultiArray& msg) {
    if (msg.data.size() < 6) return;

    geometry_msgs::PoseStamped p;
    p.header.stamp = ros::Time::now();
    p.header.frame_id = base_link_; // assume already in base/world

    p.pose.position.x = msg.data[0];
    p.pose.position.y = msg.data[1];
    p.pose.position.z = msg.data[2];

    tf2::Quaternion q;
    q.setRPY(msg.data[3], msg.data[4], msg.data[5]);
    p.pose.orientation = tf2::toMsg(q);

    std::lock_guard<std::mutex> lk(mtx_);
    obj_pose_base_ = p;
    have_obj_ = true;
  }

bool loadJointLimitsFromYaml(
    const std::string& path,
    std::map<std::string, std::pair<double,double>>& lim_out)
{
  try {
    YAML::Node root = YAML::LoadFile(path);
    auto jl = root["joint_limits"];
    if (!jl) return false;

    auto read = [&](const std::string& key) {
      return std::make_pair(
        jl[key]["min_position"].as<double>(),
        jl[key]["max_position"].as<double>());
    };

    lim_out["shoulder_pan_joint"]  = read("shoulder_pan");
    lim_out["shoulder_lift_joint"] = read("shoulder_lift");
    lim_out["elbow_joint"]         = read("elbow_joint");
    lim_out["wrist_1_joint"]       = read("wrist_1");
    lim_out["wrist_2_joint"]       = read("wrist_2");
    lim_out["wrist_3_joint"]       = read("wrist_3");

    return true;
  }
  catch (const std::exception& e) {
    ROS_ERROR("Joint limit YAML error: %s", e.what());
    return false;
  }
}



  // ------------------ KDL init ------------------
  bool initKDL() {
  ros::NodeHandle nh;
  std::string urdf;
  if (!nh.getParam(robot_desc_param_, urdf)) {
    ROS_ERROR("Param %s not found.", robot_desc_param_.c_str());
    return false;
  }

  KDL::Tree tree;
  if (!kdl_parser::treeFromString(urdf, tree)) {
    ROS_ERROR("Failed to parse URDF into KDL tree.");
    return false;
  }

  if (!tree.getChain(base_link_, ee_link_, chain_)) {
    ROS_ERROR("Failed to get KDL chain from %s to %s", base_link_.c_str(), ee_link_.c_str());
    return false;
  }

  const unsigned int nj = chain_.getNrOfJoints();
  if (nj < 6) {
    ROS_ERROR("KDL chain has %u joints (too few).", nj);
    return false;
  }

  // Load joint limits from YAML
  std::map<std::string, std::pair<double,double>> lim;
  if (!loadJointLimitsFromYaml(joint_limits_yaml_, lim)) {
    ROS_ERROR("Failed to load joint limits from %s", joint_limits_yaml_.c_str());
    return false;
  }

  q_min_ = KDL::JntArray(nj);
  q_max_ = KDL::JntArray(nj);

  unsigned int j = 0;
  for (unsigned int s = 0; s < chain_.getNrOfSegments(); ++s) {
    const KDL::Joint& kj = chain_.getSegment(s).getJoint();
    if (kj.getType() == KDL::Joint::None) continue;

    auto it = lim.find(kj.getName());
    if (it != lim.end()) {
      q_min_(j) = it->second.first;
      q_max_(j) = it->second.second;
      ROS_INFO("Limit %s: [%.3f, %.3f]", kj.getName().c_str(), q_min_(j), q_max_(j));
    } else {
      q_min_(j) = -2.0 * M_PI;
      q_max_(j) =  2.0 * M_PI;
      ROS_WARN("No YAML limit for %s, using +/-2pi", kj.getName().c_str());
    }

    j++;
    if (j >= nj) break;
  }

  fk_.reset(new KDL::ChainFkSolverPos_recursive(chain_));
  ik_vel_.reset(new KDL::ChainIkSolverVel_pinv(chain_));
  ik_pos_.reset(new KDL::ChainIkSolverPos_NR_JL(chain_, q_min_, q_max_, *fk_, *ik_vel_, 200, 1e-5));
  return true;
}


  // ------------------ IK solve ------------------
  bool solveIK(const geometry_msgs::Pose& target_pose,
               const std::vector<double>& seed,
               std::vector<double>& q_out) {
    const unsigned int nj = chain_.getNrOfJoints();
    if (seed.size() < nj) return false;

    KDL::JntArray q_seed(nj), q_res(nj);
    for (unsigned int i = 0; i < nj; ++i) q_seed(i) = seed[i];

    KDL::Frame F;
    F.p = KDL::Vector(target_pose.position.x, target_pose.position.y, target_pose.position.z);

    tf2::Quaternion q;
    tf2::fromMsg(target_pose.orientation, q);
    KDL::Rotation R = KDL::Rotation::Quaternion(q.x(), q.y(), q.z(), q.w());
    F.M = R;

    int rc = ik_pos_->CartToJnt(q_seed, F, q_res);
    if (rc < 0) return false;

    q_out.resize(nj);
    for (unsigned int i = 0; i < nj; ++i) q_out[i] = q_res(i);
    return true;
  }

  // ------------------ Motion helpers ------------------
  void publishJointTarget(const std::vector<double>& q) {
    // Your motion_planning_node expects exactly 6 joints.
    // If KDL chain returns more, take first 6.
    std_msgs::Float64MultiArray msg;
    msg.data.resize(6);
    for (int i = 0; i < 6; ++i) msg.data[i] = q[i];

    {
      std::lock_guard<std::mutex> lk(mtx_);
      ack_ = false;
    }
    pub_target_.publish(msg);
  }

  bool waitAck(double timeout_s) {
    ros::Rate r(200);
    ros::Time t0 = ros::Time::now();
    while (ros::ok()) {
      ros::spinOnce();
      r.sleep();
      bool a = false;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        a = ack_;
      }
      if (a) return true;
      if ((ros::Time::now() - t0).toSec() > timeout_s) return false;
    }
    return false;
  }

  geometry_msgs::Pose makeTopDownPose(double x, double y, double z, double yaw) {
    geometry_msgs::Pose p;
    p.position.x = x;
    p.position.y = y;
    p.position.z = z;

    double roll = fixed_roll_;
    double pitch = fixed_pitch_;
    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    p.orientation = tf2::toMsg(q);
    return p;
  }

  bool callGripper(double diameter) {
    // robust even if response has no fields
    if (!ros::service::exists(gripper_srv_name_, false)) {
      ROS_WARN("Gripper service %s not available.", gripper_srv_name_.c_str());
      return false;
    }
    ros_impedance_controller::generic_float srv;
    srv.request.data = diameter;
    if (!gripper_cli_.call(srv)) {
      ROS_WARN("Failed calling gripper service.");
      return false;
    }
    ros::Duration(gripper_wait_s_).sleep();
    return true;
  }

  // ------------------ Pick & place ------------------
  void doPickPlace(const geometry_msgs::PoseStamped& obj, std::vector<double> q_seed) {
    const double x = obj.pose.position.x;
    const double y = obj.pose.position.y;
    const double z = obj.pose.position.z;

    // Extract yaw from object orientation (optional)
    double obj_roll, obj_pitch, obj_yaw;
    {
      tf2::Quaternion q;
      tf2::fromMsg(obj.pose.orientation, q);
      tf2::Matrix3x3(q).getRPY(obj_roll, obj_pitch, obj_yaw);
    }
    const double yaw = use_only_yaw_ ? obj_yaw : obj_yaw;

    // Build EE poses
    geometry_msgs::Pose pre   = makeTopDownPose(x, y, z + z_pre_off_,   yaw);
    geometry_msgs::Pose grasp = makeTopDownPose(x, y, z + z_grasp_off_, yaw);
    geometry_msgs::Pose lift  = makeTopDownPose(x, y, z + z_lift_off_,  yaw);

    geometry_msgs::Pose place_pre = makeTopDownPose(drop_x_, drop_y_, drop_z_ + 0.10, 0.0);
    geometry_msgs::Pose place     = makeTopDownPose(drop_x_, drop_y_, drop_z_,        0.0);
    geometry_msgs::Pose retreat   = place_pre;

    std::vector<double> q_pre, q_grasp, q_lift, q_place_pre, q_place, q_retreat;

    // 1) Pregrasp
    if (!solveIK(pre, q_seed, q_pre)) {
      ROS_WARN("IK failed for pregrasp. Skipping.");
      return;
    }
    publishJointTarget(q_pre);
    if (!waitAck(ack_timeout_)) ROS_WARN("Ack timeout pregrasp.");
    q_seed = q_pre;

    // 2) Grasp
    if (!solveIK(grasp, q_seed, q_grasp)) {
      ROS_WARN("IK failed for grasp. Skipping.");
      return;
    }
    publishJointTarget(q_grasp);
    if (!waitAck(ack_timeout_)) ROS_WARN("Ack timeout grasp.");
    q_seed = q_grasp;

    // 3) Close gripper
    ROS_INFO("Closing gripper (diameter=%.1f)...", gripper_close_diam_);
    callGripper(gripper_close_diam_);

    // 4) Lift
    if (solveIK(lift, q_seed, q_lift)) {
      publishJointTarget(q_lift);
      if (!waitAck(ack_timeout_)) ROS_WARN("Ack timeout lift.");
      q_seed = q_lift;
    } else {
      ROS_WARN("IK failed for lift. Continuing anyway.");
    }

    // 5) Place pre
    if (!solveIK(place_pre, q_seed, q_place_pre)) {
      ROS_WARN("IK failed for place_pre. Aborting.");
      return;
    }
    publishJointTarget(q_place_pre);
    if (!waitAck(ack_timeout_)) ROS_WARN("Ack timeout place_pre.");
    q_seed = q_place_pre;

    // 6) Place down
    if (solveIK(place, q_seed, q_place)) {
      publishJointTarget(q_place);
      if (!waitAck(ack_timeout_)) ROS_WARN("Ack timeout place.");
      q_seed = q_place;
    } else {
      ROS_WARN("IK failed for place. Will open gripper anyway.");
    }

    // 7) Open gripper
    ROS_INFO("Opening gripper (diameter=%.1f)...", gripper_open_diam_);
    callGripper(gripper_open_diam_);

    // 8) Retreat
    if (solveIK(retreat, q_seed, q_retreat)) {
      publishJointTarget(q_retreat);
      waitAck(ack_timeout_);
    }

    ROS_INFO("Pick&place done.");
  }

  // ------------------ Members ------------------
  std::mutex mtx_;
  std::string joint_limits_yaml_;
  // topics/params
  std::string js_topic_, target_topic_, ack_topic_;
  std::string obj_pose_topic_, obj_rpy_topic_;
  std::string base_link_, ee_link_, robot_desc_param_;

  double ack_timeout_;
  double z_pre_off_, z_grasp_off_, z_lift_off_;
  double drop_x_, drop_y_, drop_z_;
  bool use_only_yaw_;
  double fixed_roll_, fixed_pitch_;

  // gripper
  std::string gripper_srv_name_;
  double gripper_open_diam_, gripper_close_diam_, gripper_wait_s_;
  ros::ServiceClient gripper_cli_;

  // ROS
  ros::Subscriber sub_js_, sub_ack_;
  ros::Subscriber sub_obj_pose_, sub_obj_rpy_;
  ros::Publisher pub_target_;

  // state
  bool have_js_;
  bool have_obj_;
  bool ack_;
  std::vector<double> q_cur_;
  std::vector<int> idx_;
  std::vector<std::string> ur5_names_;
  geometry_msgs::PoseStamped obj_pose_base_;

  // KDL
  bool ik_ready_ = false;
  KDL::Chain chain_;
  KDL::JntArray q_min_, q_max_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;
  std::unique_ptr<KDL::ChainIkSolverVel_pinv> ik_vel_;
  std::unique_ptr<KDL::ChainIkSolverPos_NR_JL> ik_pos_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "pick_place_ik");
  PickPlaceIK n;
  n.spin();
  return 0;
}

