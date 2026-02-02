/**
* @file task_planning_node.cpp
* @brief Task-level pick-and-place planning with inverse kinematics.
*
* This ROS node implements a complete pick-and-place pipeline for a UR5
* robotic arm. It receives object poses from the perception system,
* computes feasible grasp configurations using inverse kinematics,
* and generates a sequence of joint-space targets to execute grasp,
* transport, and placement motions.
*
* The node performs decision making at task level, including:
* - Grasp orientation selection
* - Singularity and joint-jump avoidance
* - Class-dependent placement strategy
* - Motion synchronization via acknowledgements
*
* Inverse kinematics is solved using the KDL library, while trajectory
* execution is delegated to a dedicated motion execution node.
*/

#include <ros/ros.h>
#include <ros/package.h>
#include <yaml-cpp/yaml.h>

#include <sensor_msgs/JointState.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolverpos_nr.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>

#include <regex>
#include <map>
#include <mutex>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <memory>
#include <array>

/**
* @class PickPlaceIK
* @brief Task planner for vision-based pick-and-place.
*
* This class coordinates perception, inverse kinematics, and motion
* execution to perform autonomous pick-and-place operations.
* It generates a sequence of joint-space goals based on object pose,
* orientation, and semantic class, and synchronizes execution using
* acknowledgement messages.
*/
class PickPlaceIK {
public:
  /**
  * @brief Initialize the pick-and-place task planner.
  *
  * Loads configuration parameters, initializes ROS interfaces,
  * sets up the KDL kinematic chain, and prepares internal state
  * for task execution.
  */
  PickPlaceIK()
  : have_js_(false), have_obj_(false), ack_(false)
  {
    ros::NodeHandle nh, pnh("~");

    const std::string ur_pkg = ros::package::getPath("ur_description");
    joint_limits_yaml_ = ur_pkg + "/config/ur5e/joint_limits.yaml";

    pnh.param<std::string>("joint_states_topic", js_topic_, "/ur5/joint_states");
    pnh.param<std::string>("joint_target_topic", target_topic_, "/ur5/joint_target");
    pnh.param<std::string>("ack_topic", ack_topic_, "/acknowledgement");

    pnh.param<std::string>("object_pose_topic", obj_pose_topic_, "/vision/object_pose");
    pnh.param<std::string>("object_rpy_topic",  obj_rpy_topic_,  "/vision/object_rpy");

    pnh.param<std::string>("base_link", base_link_, "base_link");
    pnh.param<std::string>("ee_link", ee_link_, "tool0");
    pnh.param<std::string>("robot_description_param", robot_desc_param_, "/ur5/robot_description");

    pnh.param<std::string>("object_name_topic", obj_name_topic_, "/vision/object_name");
    sub_obj_name_ = nh.subscribe(obj_name_topic_, 1, &PickPlaceIK::objNameCb, this);

    pnh.param<std::string>("object_uid_topic", obj_uid_topic_, "/vision/object_uid");
    sub_obj_uid_ = nh.subscribe(obj_uid_topic_, 1, &PickPlaceIK::objUidCb, this);

    pnh.param("ack_timeout", ack_timeout_, 10.0);

    pnh.param("z_pre",   z_pre_off_,   0.12);
    pnh.param("z_grasp", z_grasp_off_, -0.016);
    pnh.param("z_lift",  z_lift_off_,  0.20);

    pnh.param("table_z", table_z_, -0.85);
    pnh.param("z_clear", z_clear_, 0.005);

    pnh.param("drop_x", drop_x_, 0.40);
    pnh.param("drop_y", drop_y_, 0.20);
    pnh.param("drop_z", drop_z_, -0.83);

    pnh.param("place_pre_up", place_pre_up_, 0.28);
    pnh.param("safe_up_after_open", safe_up_after_open_, 0.06);

    pnh.param("ik_max_jump", ik_max_jump_, 2.5);
    pnh.param("min_xy_radius", min_xy_radius_, 0.12);

    pnh.param("fixed_roll",  fixed_roll_,  M_PI);
    pnh.param("fixed_pitch", fixed_pitch_, 0.0);

    pnh.param("hand_open",  hand_open_,  0.85);
    pnh.param("hand_close", hand_close_, 0.0);

    pnh.param("grasp_settle_s", grasp_settle_s_, 0.55);
    pnh.param("use_home_after_place", use_home_after_place_, false);

    loadHomeJoints(pnh);

    sub_js_       = nh.subscribe(js_topic_, 1, &PickPlaceIK::jsCb, this);
    sub_ack_      = nh.subscribe(ack_topic_, 1, &PickPlaceIK::ackCb, this);
    sub_obj_pose_ = nh.subscribe(obj_pose_topic_, 1, &PickPlaceIK::objPoseCb, this);
    sub_obj_rpy_  = nh.subscribe(obj_rpy_topic_,  1, &PickPlaceIK::objRpyCb,  this);

    pub_target_ = nh.advertise<std_msgs::Float64MultiArray>(target_topic_, 1);

    pub_req_ = nh.advertise<std_msgs::Bool>("/vision/request_object", 1, true);
    last_req_ = ros::Time(0);

    ur5_names_ = {
      "shoulder_pan_joint",
      "shoulder_lift_joint",
      "elbow_joint",
      "wrist_1_joint",
      "wrist_2_joint",
      "wrist_3_joint",
      "hand_1_joint",
      "hand_2_joint"
    };

    q_gripper_cmd_[0] = hand_open_;
    q_gripper_cmd_[1] = hand_open_;

    ik_ready_ = initKDL();
    if (!ik_ready_) {
      ROS_ERROR("KDL IK init FAILED.");
    } else {
      ROS_INFO("KDL IK ready. base=%s ee=%s", base_link_.c_str(), ee_link_.c_str());
    }

    ROS_INFO("PickPlaceIK ready:");
    ROS_INFO("  drop: x=%.3f y=%.3f z=%.3f", drop_x_, drop_y_, drop_z_);
    ROS_INFO("  table_z=%.3f z_clear=%.3f", table_z_, z_clear_);
    ROS_INFO("  z_pre=%.3f  z_grasp=%.3f  z_lift=%.3f", z_pre_off_, z_grasp_off_, z_lift_off_);
    ROS_INFO("  hand_open=%.3f hand_close=%.3f", hand_open_, hand_close_);
  }

  /**
  * @brief Main task execution loop.
  *
  * Continuously monitors perception input and triggers a complete
  * pick-and-place sequence when a new object is detected.
  * The loop also handles object request signaling and duplicate
  * execution prevention.
  */
  void spin() {
    ros::Rate r(50);
    while (ros::ok()) {
      ros::spinOnce();

      geometry_msgs::PoseStamped obj;
      std::vector<double> q_seed8_raw;
      std::string uid;
      bool need_request = false;

      {
        std::lock_guard<std::mutex> lk(mtx_);

        if (!ik_ready_ || !have_js_) { r.sleep(); continue; }

        if (!have_obj_) {
          need_request = true;
        } else {
          obj = obj_pose_base_;
          q_seed8_raw = q_cur_raw_;
          uid = current_uid_;
          have_obj_ = false;

          if (!uid.empty() && uid == last_uid_done_) {
            need_request = true;
          }
        }
      }

      if (need_request) {
        if ((ros::Time::now() - last_req_).toSec() > 0.5) { // max 2 Hz
          std_msgs::Bool req; req.data = true;
          pub_req_.publish(req);
          last_req_ = ros::Time::now();
        }
        r.sleep();
        continue;
      }

      doPickPlace(obj, q_seed8_raw);

      {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!uid.empty()) last_uid_done_ = uid;
      }

      r.sleep();
    }
  }

private:
  // -------------------- small utils --------------------
  static double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
  }

  static double wrapPi(double a) {
    while (a >  M_PI) a -= 2.0*M_PI;
    while (a < -M_PI) a += 2.0*M_PI;
    return a;
  }

  static double wrapNear(double q_small, double q_ref_raw) {
    double dq = q_small - q_ref_raw;
    while (dq >  M_PI) dq -= 2.0*M_PI;
    while (dq < -M_PI) dq += 2.0*M_PI;
    return q_ref_raw + dq;
  }

  static double snap90(double yaw) {
    const double step = M_PI_2;
    return wrapPi(std::round(wrapPi(yaw) / step) * step);
  }

  static bool parseXYZ(const std::string& s, int& X, int& Y, int& Z) {
    std::regex r("X(\\d+)-Y(\\d+)-Z(\\d+)");
    std::smatch m;
    if (!std::regex_search(s, m, r)) return false;
    X = std::stoi(m[1].str());
    Y = std::stoi(m[2].str());
    Z = std::stoi(m[3].str());
    return true;
  }

  static int classIdFromXYZ(int X, int Y, int Z) {
    if (X==1 && Y==1 && Z==2) return 0;
    if (X==1 && Y==2 && Z==1) return 1;
    if (X==1 && Y==2 && Z==2) return 2;
    if (X==1 && Y==3 && Z==2) return 3;
    if (X==1 && Y==4 && Z==1) return 4;
    if (X==1 && Y==4 && Z==2) return 5;
    if (X==2 && Y==2 && Z==2) return 6;
    return -1;
  }

  static double jumpNorm6(const std::vector<double>& q6_raw, const std::vector<double>& seed8_raw) {
    const int n = std::min(6, (int)seed8_raw.size());
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
      const double d = wrapPi(q6_raw[i] - seed8_raw[i]);
      s += d*d;
    }
    return std::sqrt(s);
  }

  static bool nearWristSingularity(const std::vector<double>& q6_raw) {
    if (q6_raw.size() < 5) return false;
    const double w2 = wrapPi(q6_raw[4]);  // wrist_2
    return (std::abs(std::sin(w2)) < 0.18);
  }

  static std::vector<double> makeSeed8(const std::vector<double>& q6_raw, const std::array<double,2>& grip) {
    std::vector<double> s(8, 0.0);
    for (int i = 0; i < 6; ++i) s[i] = q6_raw[i];
    s[6] = grip[0];
    s[7] = grip[1];
    return s;
  }

  geometry_msgs::Pose makeTopDownPose(double x, double y, double z, double yaw) {
    geometry_msgs::Pose p;
    p.position.x = x;
    p.position.y = y;
    p.position.z = z;
    tf2::Quaternion q;
    q.setRPY(fixed_roll_, fixed_pitch_, yaw);
    p.orientation = tf2::toMsg(q);
    return p;
  }

  // -------------------- ROS callbacks --------------------
  void objNameCb(const std_msgs::String& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    last_obj_name_ = msg.data;
  }

  void objUidCb(const std_msgs::String& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    current_uid_ = msg.data;
  }

  /**
  * @brief Joint state callback.
  *
  * Updates the current joint configuration used as IK seed.
  */
  void jsCb(const sensor_msgs::JointState& msg) {
    if (msg.name.size() != msg.position.size()) return;
    std::lock_guard<std::mutex> lk(mtx_);

    if (idx_.empty()) {
      idx_.assign((int)ur5_names_.size(), -1);
      for (int k = 0; k < (int)ur5_names_.size(); ++k) {
        for (int i = 0; i < (int)msg.name.size(); ++i) {
          if (msg.name[i] == ur5_names_[k]) { idx_[k] = i; break; }
        }
      }
      bool ok = true;
      for (int k = 0; k < (int)ur5_names_.size(); ++k) ok = ok && (idx_[k] >= 0);
      if (!ok) {
        ROS_ERROR("UR5 joint names not found in /joint_states.");
        return;
      }
      ROS_INFO("Joint index map created from /joint_states.");
    }

    q_cur_raw_.resize(8);
    for (int k = 0; k < 6; ++k) q_cur_raw_[k] = msg.position[idx_[k]];
    q_cur_raw_[6] = clamp(msg.position[idx_[6]], hand_close_, hand_open_);
    q_cur_raw_[7] = clamp(msg.position[idx_[7]], hand_close_, hand_open_);

    have_js_ = true;
  }

  void ackCb(const std_msgs::Bool& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    ack_ = msg.data;
  }

  /**
  * @brief Receive object pose from perception system.
  */
  void objPoseCb(const geometry_msgs::PoseStamped& msg) {
    if (msg.header.frame_id != base_link_) return;
    std::lock_guard<std::mutex> lk(mtx_);
    obj_pose_base_ = msg;
    have_obj_ = true;
  }

  void objRpyCb(const std_msgs::Float64MultiArray& msg) {
    if (msg.data.size() < 6) return;

    geometry_msgs::PoseStamped out;
    out.header.stamp = ros::Time::now();
    out.header.frame_id = base_link_;

    out.pose.position.x = msg.data[0];
    out.pose.position.y = msg.data[1];
    out.pose.position.z = msg.data[2];

    tf2::Quaternion q;
    q.setRPY(msg.data[3], msg.data[4], msg.data[5]);
    out.pose.orientation = tf2::toMsg(q);

    std::lock_guard<std::mutex> lk(mtx_);
    obj_pose_base_ = out;
    have_obj_ = true;
  }

  // -------------------- ack + publish --------------------
  bool waitAck(double timeout_s) {
    ros::Rate r(200);
    const ros::Time t0 = ros::Time::now();
    while (ros::ok()) {
      ros::spinOnce();
      r.sleep();
      bool a = false;
      { std::lock_guard<std::mutex> lk(mtx_); a = ack_; }
      if (a) return true;
      if ((ros::Time::now() - t0).toSec() > timeout_s) return false;
    }
    return false;
  }

  bool waitGripperAt(double cmd, double tol, double timeout_s) {
    ros::Rate r(200);
    const ros::Time t0 = ros::Time::now();
    while (ros::ok()) {
      ros::spinOnce();
      double g1=0, g2=0;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        if (q_cur_raw_.size() >= 8) { g1 = q_cur_raw_[6]; g2 = q_cur_raw_[7]; }
      }
      if (std::abs(g1 - cmd) < tol && std::abs(g2 - cmd) < tol) return true;
      if ((ros::Time::now() - t0).toSec() > timeout_s) return false;
      r.sleep();
    }
    return false;
  }

  void publishJointTarget(const std::vector<double>& q_arm6_raw) {
    if (q_arm6_raw.size() < 6) return;

    std::vector<double> qref8_raw;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      qref8_raw = q_cur_raw_;
      ack_ = false;
    }

    std_msgs::Float64MultiArray msg;
    msg.data.resize(8);

    for (int i = 0; i < 6; ++i) {
      const double q_small = wrapPi(q_arm6_raw[i]);
      msg.data[i] = wrapNear(q_small, qref8_raw[i]);
    }
    msg.data[6] = q_gripper_cmd_[0];
    msg.data[7] = q_gripper_cmd_[1];
    pub_target_.publish(msg);
  }

  // -------------------- home --------------------
  void loadHomeJoints(ros::NodeHandle& pnh) {
    std::vector<double> q0;
    bool ok_q0 = ros::param::get("/ur5/q_0", q0) || ros::param::get("/ur5e/q_0", q0);
    if (!ok_q0) pnh.getParam("home_joints", q0);

    if (q0.size() != 6) {
      q0 = {0.0, -1.57, 1.57, -1.57, -1.57, 0.0};
      ROS_WARN("Home joints not found. Using fallback home.");
    }
    home_q_ = q0;
  }

  // -------------------- KDL --------------------
  bool loadJointLimitsFromYaml(const std::string& path,
                               std::map<std::string, std::pair<double,double>>& lim_out) {
    try {
      YAML::Node root = YAML::LoadFile(path);
      auto jl = root["joint_limits"];
      if (!jl) return false;

      auto read = [&](const std::string& key, double& mn, double& mx) -> bool {
        if (!jl[key]) return false;
        auto n = jl[key];
        if (n["min_position"] && n["max_position"]) {
          mn = n["min_position"].as<double>();
          mx = n["max_position"].as<double>();
          return true;
        }
        if (n["min"] && n["max"]) {
          mn = n["min"].as<double>();
          mx = n["max"].as<double>();
          return true;
        }
        return false;
      };

      double mn, mx;
      if (read("shoulder_pan", mn, mx))   lim_out["shoulder_pan_joint"]   = {mn, mx};
      if (read("shoulder_lift", mn, mx))  lim_out["shoulder_lift_joint"]  = {mn, mx};
      if (read("elbow_joint", mn, mx))    lim_out["elbow_joint"]          = {mn, mx};
      if (read("wrist_1", mn, mx))        lim_out["wrist_1_joint"]        = {mn, mx};
      if (read("wrist_2", mn, mx))        lim_out["wrist_2_joint"]        = {mn, mx};
      if (read("wrist_3", mn, mx))        lim_out["wrist_3_joint"]        = {mn, mx};

      return !lim_out.empty();
    } catch (...) {
      return false;
    }
  }

  bool initKDL() {
    ros::NodeHandle nh;
    std::string urdf;
    if (!nh.getParam(robot_desc_param_, urdf)) return false;

    KDL::Tree tree;
    if (!kdl_parser::treeFromString(urdf, tree)) return false;
    if (!tree.getChain(base_link_, ee_link_, chain_)) return false;

    const unsigned int nj = chain_.getNrOfJoints();
    if (nj < 6) return false;

    q_min_ = KDL::JntArray(nj);
    q_max_ = KDL::JntArray(nj);
    for (unsigned int j = 0; j < nj; ++j) {
      q_min_(j) = -2.0*M_PI;
      q_max_(j) =  2.0*M_PI;
    }

    std::map<std::string, std::pair<double,double>> lim;
    (void)loadJointLimitsFromYaml(joint_limits_yaml_, lim);

    fk_.reset(new KDL::ChainFkSolverPos_recursive(chain_));
    ik_vel_.reset(new KDL::ChainIkSolverVel_pinv(chain_));
    ik_pos_.reset(new KDL::ChainIkSolverPos_NR_JL(chain_, q_min_, q_max_, *fk_, *ik_vel_, 200, 1e-5));
    ik_pos_nolimit_.reset(new KDL::ChainIkSolverPos_NR(chain_, *fk_, *ik_vel_, 200, 1e-5));
    return true;
  }

  /**
  * @brief Solve inverse kinematics for a desired end-effector pose.
  *
  * Uses KDL numerical solvers to compute a joint configuration
  * consistent with the target pose, preserving continuity with
  * the provided seed configuration.
  *
  * @param target_pose Desired end-effector pose.
  * @param seed8_raw Seed joint configuration.
  * @param q_out6_raw Resulting arm joint configuration.
  * @return True if a valid solution is found.
  */
  bool solveIK(const geometry_msgs::Pose& target_pose,
               const std::vector<double>& seed8_raw,
               std::vector<double>& q_out6_raw) {

    const unsigned int nj = chain_.getNrOfJoints();
    KDL::JntArray q_seed(nj), q_res(nj);

    for (unsigned int i = 0; i < nj; ++i) {
      const double raw = (i < seed8_raw.size()) ? seed8_raw[i] : 0.0;
      q_seed(i) = wrapPi(raw);
    }

    KDL::Frame F;
    F.p = KDL::Vector(target_pose.position.x, target_pose.position.y, target_pose.position.z);
    tf2::Quaternion q;
    tf2::fromMsg(target_pose.orientation, q);
    F.M = KDL::Rotation::Quaternion(q.x(), q.y(), q.z(), q.w());

    int rc = ik_pos_->CartToJnt(q_seed, F, q_res);
    if (rc < 0) {
      rc = ik_pos_nolimit_->CartToJnt(q_seed, F, q_res);
      if (rc < 0) return false;
    }

    q_out6_raw.resize(nj);
    for (unsigned int i = 0; i < nj; ++i) {
      const double q_small = wrapPi(q_res(i));
      const double raw_ref = (i < seed8_raw.size()) ? seed8_raw[i] : 0.0;
      q_out6_raw[i] = wrapNear(q_small, raw_ref);
    }
    return true;
  }

  /**
  * @brief Select the best grasp yaw among candidate orientations.
  *
  * Evaluates multiple yaw hypotheses and selects the one minimizing
  * joint displacement while avoiding wrist singularities.
  */
  bool solveBestYaw(double px, double py, double pz,
                    double yaw0,
                    const std::vector<double>& seed8_raw,
                    std::vector<double>& qbest_raw,
                    double& best_jump) {

    const double cands[2] = { wrapPi(yaw0), wrapPi(yaw0 + M_PI) };

    bool any = false;
    best_jump = 1e9;
    std::vector<double> qtmp_raw;

    for (int i = 0; i < 2; ++i) {
      geometry_msgs::Pose p = makeTopDownPose(px, py, pz, cands[i]);
      if (!solveIK(p, seed8_raw, qtmp_raw)) continue;
      if (nearWristSingularity(qtmp_raw)) continue;

      const double j = jumpNorm6(qtmp_raw, seed8_raw);
      if (j < best_jump) {
        best_jump = j;
        qbest_raw = qtmp_raw;
        any = true;
      }
    }
    return any;
  }

  // -------------------- motion helpers (CLASS METHODS!) --------------------
  /**
  * @brief Move the robot to a Cartesian target selecting the best yaw.
  *
  * Generates IK solutions for multiple yaw candidates, evaluates their
  * feasibility, and commands the motion with the lowest cost.
  *
  * @return True if a valid motion is executed.
  */
  bool tryMoveBestYaw(double px, double py, double pz, double yaw_hint,
                      const char* tag, double max_jump,
                      std::vector<double>& q_seed8_raw)
  {
    const double cands[4] = {
      snap90(yaw_hint),
      snap90(yaw_hint + M_PI_2),
      snap90(yaw_hint - M_PI_2),
      snap90(yaw_hint + M_PI)
    };

    bool ok = false;
    double best_cost = 1e18;
    std::vector<double> best_q6;

    for (int i = 0; i < 4; ++i) {
      std::vector<double> q6;
      double bestJ = 0.0;

      if (!solveBestYaw(px, py, pz, cands[i], q_seed8_raw, q6, bestJ)) continue;
      if (bestJ > max_jump) continue;
      if (nearWristSingularity(q6)) continue;

      const double d0 = wrapPi(q6[0] - q_seed8_raw[0]);  // shoulder pan delta
      const double w2 = wrapPi(q6[4]);                   // wrist_2
      const double wrist_pen = 1.0 / (std::fabs(std::sin(w2)) + 0.08);

      const double cost = bestJ + 1.3*(d0*d0) + 0.55*wrist_pen;

      if (cost < best_cost) {
        best_cost = cost;
        best_q6 = q6;
        ok = true;
      }
    }

    if (!ok) {
      ROS_WARN("IK failed for all yaw candidates at %s.", tag);
      return false;
    }

    ROS_INFO("Move %s (cost=%.3f)", tag, best_cost);
    publishJointTarget(best_q6);
    if (!waitAck(ack_timeout_)) {
      ROS_WARN("Ack timeout %s.", tag);
      return false;
    }

    q_seed8_raw = makeSeed8(best_q6, q_gripper_cmd_);
    return true;
  }

  /**
  * @brief Move the robot to a Cartesian target with fixed yaw.
  *
  * Executes a motion only if the requested orientation is feasible
  * and safe, without fallback yaw alternatives.
  */
  bool tryMoveFixedYaw(double px, double py, double pz, double yaw,
                       const char* tag, double max_jump,
                       std::vector<double>& q_seed8_raw)
  {
    std::vector<double> q6_raw;
    geometry_msgs::Pose p = makeTopDownPose(px, py, pz, yaw);

    if (!solveIK(p, q_seed8_raw, q6_raw)) {
      ROS_WARN("IK failed %s (fixed yaw).", tag);
      return false;
    }
    if (nearWristSingularity(q6_raw)) {
      ROS_WARN("Near wrist singularity at %s (fixed yaw).", tag);
      return false;
    }

    const double j = jumpNorm6(q6_raw, q_seed8_raw);
    if (j > max_jump) {
      ROS_WARN("IK jump too large at %s: %.3f > %.3f.", tag, j, max_jump);
      return false;
    }

    const double d0 = wrapPi(q6_raw[0] - q_seed8_raw[0]);
    if ((d0*d0) > 2.2) {
      ROS_WARN("Shoulder flip too big at %s (d0=%.3f). Reject.", tag, d0);
      return false;
    }

    ROS_INFO("Move %s (jump=%.3f)", tag, j);
    publishJointTarget(q6_raw);
    if (!waitAck(ack_timeout_)) {
      ROS_WARN("Ack timeout %s.", tag);
      return false;
    }

    q_seed8_raw = makeSeed8(q6_raw, q_gripper_cmd_);
    return true;
  }

  // -------------------- MAIN pick&place --------------------
  /**
  * @brief Execute a full pick-and-place sequence for a detected object.
  *
  * This method computes grasp and placement poses, selects suitable
  * end-effector orientations, solves inverse kinematics, and issues
  * a sequence of joint-space motion commands to grasp, lift, transport,
  * and place the object.
  *
  * Safety checks include joint jump limits, wrist singularity avoidance,
  * minimum reach constraints, and table clearance.
  *
  * @param obj Object pose expressed in the robot base frame.
  * @param q_seed8_raw Initial joint configuration used as IK seed.
  */
  void doPickPlace(const geometry_msgs::PoseStamped& obj, std::vector<double> q_seed8_raw)
  {
    auto doOpen = [&](const char* tag)->void {
      ROS_INFO("Opening gripper (%s) (g=%.3f)", tag, hand_open_);
      q_gripper_cmd_[0] = hand_open_;
      q_gripper_cmd_[1] = hand_open_;
      std::vector<double> q6_raw(6);
      for (int i = 0; i < 6; ++i) q6_raw[i] = q_seed8_raw[i];
      publishJointTarget(q6_raw);
      waitAck(ack_timeout_);
      waitGripperAt(hand_open_, 0.02, 1.2);
      ros::Duration(0.05).sleep();
    };

    auto doClose = [&](double cmd, const char* tag)->void {
      ROS_INFO("Closing gripper (%s) (g=%.3f)", tag, cmd);
      q_gripper_cmd_[0] = cmd;
      q_gripper_cmd_[1] = cmd;
      std::vector<double> q6_raw(6);
      for (int i = 0; i < 6; ++i) q6_raw[i] = q_seed8_raw[i];
      publishJointTarget(q6_raw);
      waitAck(ack_timeout_);
      ros::Duration(grasp_settle_s_).sleep();
    };

    auto retreatSafe = [&](double z_safe, double yaw)->void {
      const double z_up = z_safe + 0.28;
      (void)tryMoveBestYaw(0.22, 0.20, z_up, yaw, "retreat_safe", 3.8, q_seed8_raw);
    };

    // obj
    const double x_in = obj.pose.position.x;
    const double y_in = obj.pose.position.y;
    const double z_in = obj.pose.position.z;

    std::string obj_name;
    { std::lock_guard<std::mutex> lk(mtx_); obj_name = last_obj_name_; }

    int Xc=0, Yc=0, Zc=0;
    int class_id = -1;
    if (parseXYZ(obj_name, Xc, Yc, Zc)) class_id = classIdFromXYZ(Xc, Yc, Zc);

    double rr=0, pp=0, yaw_obj=0;
    {
      tf2::Quaternion qtmp;
      tf2::fromMsg(obj.pose.orientation, qtmp);
      tf2::Matrix3x3(qtmp).getRPY(rr, pp, yaw_obj);
    }
    yaw_obj = snap90(yaw_obj);

    const double rxy = std::sqrt(x_in*x_in + y_in*y_in);
    if (rxy < min_xy_radius_) {
      ROS_WARN("Object too close to base: r=%.3f < %.3f. Skipping.", rxy, min_xy_radius_);
      return;
    }

    const double z_min  = table_z_ + z_clear_;
    const double z_safe = std::max(z_in, z_min);
    const double drop_z_safe = std::max(drop_z_, z_min);

    double z_pre_high = z_safe + z_pre_off_ + 0.10;
    double z_pre      = z_safe + z_pre_off_;
    double z_grasp    = z_safe + z_grasp_off_;
    z_grasp -= 0.006;

    const double z_grasp_min = z_min - 0.010;
    const double z_pre_min   = z_min + 0.015;
    z_grasp = std::max(z_grasp, z_grasp_min);
    z_pre   = std::max(z_pre,   z_pre_min);

    double close_cmd_first = hand_close_;
    double close_cmd_final = hand_close_;
    if (class_id == 6) { close_cmd_first = 0.20; close_cmd_final = 0.10; }

    auto dropYForClass = [&](int cid)->double {
      switch (cid) {
        case 6: return  0.40;
        case 5: return  0.32;
        case 2: return  0.24;
        case 3: return  0.16;
        case 4: return  0.07;
        case 1: return  0.00;
        case 0: return -0.07;
        default: return drop_y_;
      }
    };

    const double drop_x_eff = drop_x_;
    const double drop_y_eff = dropYForClass(class_id);

    // choose yaw_grasp: yaw_obj or yaw_obj+90 by IK on z_pre
    double yaw_grasp = yaw_obj;
    {
      struct Cand { double yaw; double jump; };
      std::vector<Cand> good;
      auto eval = [&](double yaw_try){
        std::vector<double> q6;
        double bestJ = 0.0;
        if (!solveBestYaw(x_in, y_in, z_pre, yaw_try, q_seed8_raw, q6, bestJ)) return;
        if (nearWristSingularity(q6)) return;
        good.push_back({snap90(yaw_try), bestJ});
      };
      eval(yaw_obj);
      eval(wrapPi(yaw_obj + M_PI_2));
      if (!good.empty()) {
        std::sort(good.begin(), good.end(), [](const Cand& a, const Cand& b){ return a.jump < b.jump; });
        yaw_grasp = good.front().yaw;
      }
    }

    const double yaw_place = 0.0; // richiesto

    ROS_WARN("OBJ x=%.3f y=%.3f z=%.3f yaw_obj=%.3f yaw_grasp=%.3f yaw_place=%.3f class=%d name=%s -> DROP (%.3f,%.3f)",
             x_in, y_in, z_in, yaw_obj, yaw_grasp, yaw_place, class_id, obj_name.c_str(), drop_x_eff, drop_y_eff);

    // open
    doOpen("pre");

    // staging vicino (evita salita/flips)
    {
      const double x_stage = 0.22;
      const double y_stage = clamp(y_in, 0.12, 0.28);
      const double z_stage = z_safe + 0.24;
      (void)tryMoveBestYaw(x_stage, y_stage, z_stage, yaw_grasp, "start_stage", 3.2, q_seed8_raw);
    }

    // approach
    double x = x_in, y = y_in;

    if (!tryMoveBestYaw(x, y, z_pre_high, yaw_grasp, "pregrasp_high", 3.4, q_seed8_raw)) { retreatSafe(z_safe, yaw_grasp); return; }
    if (!tryMoveBestYaw(x, y, z_pre,      yaw_grasp, "pregrasp",      3.2, q_seed8_raw)) { retreatSafe(z_safe, yaw_grasp); return; }
    if (!tryMoveBestYaw(x, y, z_grasp,    yaw_grasp, "grasp",         3.2, q_seed8_raw)) { retreatSafe(z_safe, yaw_grasp); return; }

    // micro deeper
    {
      const double z_deeper = std::max(z_grasp - 0.004, z_grasp_min);
      if (z_deeper < z_grasp - 1e-4) {
        (void)tryMoveBestYaw(x, y, z_deeper, yaw_grasp, "grasp_deeper", 3.2, q_seed8_raw);
      }
    }

    doClose(close_cmd_first, "hold1");
    if (class_id == 6) doClose(close_cmd_final, "hold2_final");

    // lift/carry
    const double z_detach = z_safe + 0.11;
    const double z_carry  = z_safe + 0.28;

    if (!tryMoveBestYaw(x, y, z_detach, yaw_grasp, "detach_up", 3.8, q_seed8_raw)) { retreatSafe(z_safe, yaw_grasp); return; }
    if (!tryMoveBestYaw(x, y, z_carry,  yaw_grasp, "carry_up",  3.8, q_seed8_raw)) { retreatSafe(z_safe, yaw_grasp); return; }

    // lane
    const double y_lane = 0.20;

    if (!tryMoveBestYaw(0.18, y,      z_carry, yaw_grasp, "carry_forward_safe", 3.6, q_seed8_raw)) { retreatSafe(z_safe, yaw_grasp); return; }
    if (!tryMoveBestYaw(0.18, y_lane, z_carry, yaw_grasp, "carry_to_lane",      3.6, q_seed8_raw)) { retreatSafe(z_safe, yaw_grasp); return; }
    if (!tryMoveBestYaw(drop_x_eff, y_lane,     z_carry, yaw_grasp, "carry_lane_safeY",     3.8, q_seed8_raw)) { retreatSafe(z_safe, yaw_grasp); return; }
    if (!tryMoveBestYaw(drop_x_eff, drop_y_eff, z_carry, yaw_grasp, "carry_lane_to_dropY",  3.8, q_seed8_raw)) { retreatSafe(z_safe, yaw_grasp); return; }

    // align yaw=0 (no fallback)
    if (!tryMoveFixedYaw(drop_x_eff, drop_y_eff, z_carry, yaw_place, "yaw_align_place_0", 4.2, q_seed8_raw)) {
      ROS_WARN("Cannot align to yaw=0 -> abort to keep yaw=0 constraint");
      retreatSafe(z_safe, yaw_grasp);
      return;
    }

    // PLACE: scende, poi apre
    const double z_place_pre = drop_z_safe + place_pre_up_;
    if (!tryMoveFixedYaw(drop_x_eff, drop_y_eff, z_place_pre, yaw_place, "place_pre", 4.2, q_seed8_raw)) {
      retreatSafe(z_safe, yaw_grasp);
      return;
    }

    const double z_place = drop_z_safe + 0.050;
    if (!tryMoveFixedYaw(drop_x_eff, drop_y_eff, z_place, yaw_place, "place", 4.2, q_seed8_raw)) {
      retreatSafe(z_safe, yaw_grasp);
      return;
    }

    ros::Duration(0.10).sleep();
    doOpen("release");

    (void)tryMoveFixedYaw(drop_x_eff, drop_y_eff, z_place + 0.140, yaw_place, "detach_up_after_release", 4.5, q_seed8_raw);
    (void)tryMoveFixedYaw(drop_x_eff, y_lane,     z_carry,         yaw_place, "retreat_lane",           4.5, q_seed8_raw);

    ROS_INFO("Pick&place done. class=%d name=%s pick=(%.3f,%.3f) drop=(%.3f,%.3f) yaw_grasp=%.3f yaw_place=%.3f",
             class_id, obj_name.c_str(), x, y, drop_x_eff, drop_y_eff, yaw_grasp, yaw_place);
  }

  // ---------- Members ----------
  std::mutex mtx_;

  ros::Publisher pub_req_;
  ros::Time last_req_;

  std::string obj_name_topic_;
  ros::Subscriber sub_obj_name_;
  std::string last_obj_name_;

  std::string obj_uid_topic_;
  ros::Subscriber sub_obj_uid_;
  std::string current_uid_;
  std::string last_uid_done_;

  std::string joint_limits_yaml_;

  double min_xy_radius_ = 0.12;
  double ik_max_jump_ = 2.5;

  std::string js_topic_, target_topic_, ack_topic_;
  std::string obj_pose_topic_, obj_rpy_topic_;
  std::string base_link_, ee_link_, robot_desc_param_;

  double ack_timeout_;
  double z_pre_off_, z_grasp_off_, z_lift_off_;
  double drop_x_, drop_y_, drop_z_;
  double place_pre_up_, safe_up_after_open_;
  double fixed_roll_, fixed_pitch_;

  double table_z_, z_clear_;
  double hand_open_, hand_close_;
  double grasp_settle_s_;

  std::array<double,2> q_gripper_cmd_{0.8, 0.8};

  bool use_home_after_place_;
  std::vector<double> home_q_;

  ros::Subscriber sub_js_, sub_ack_;
  ros::Subscriber sub_obj_pose_, sub_obj_rpy_;
  ros::Publisher pub_target_;

  bool have_js_;
  bool have_obj_;
  bool ack_;

  std::vector<double> q_cur_raw_;
  std::vector<int> idx_;
  std::vector<std::string> ur5_names_;
  geometry_msgs::PoseStamped obj_pose_base_;

  bool ik_ready_ = false;

  KDL::Chain chain_;
  KDL::JntArray q_min_, q_max_;

  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;
  std::unique_ptr<KDL::ChainIkSolverVel_pinv> ik_vel_;
  std::unique_ptr<KDL::ChainIkSolverPos_NR_JL> ik_pos_;
  std::unique_ptr<KDL::ChainIkSolverPos_NR>    ik_pos_nolimit_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "task_planning_node");
  PickPlaceIK n;
  n.spin();
  return 0;
}
