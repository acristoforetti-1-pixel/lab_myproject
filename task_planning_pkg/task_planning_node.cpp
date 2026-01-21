#include <ros/ros.h>
#include <ros/package.h>
#include <yaml-cpp/yaml.h>

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
#include <kdl/chainiksolverpos_nr.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>

#include <std_msgs/String.h>
#include <regex>
#include <map>
#include <mutex>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <memory>
#include <array>

class PickPlaceIK {
public:
  PickPlaceIK()
  : have_js_(false),
    have_obj_(false),
    ack_(false)
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
    pnh.param("ack_timeout", ack_timeout_, 10.0);

    pnh.param("z_pre",   z_pre_off_,   0.12);
    //  scendi 4mm sotto lo "z_safe" per prendere anche i piccoli
    pnh.param("z_grasp", z_grasp_off_, -0.016);
    pnh.param("z_lift",  z_lift_off_,  0.20);

    pnh.param("table_z", table_z_, -0.85);
    pnh.param("z_clear", z_clear_, 0.005);

    pnh.param("drop_x", drop_x_, 0.40);
    pnh.param("drop_y", drop_y_, 0.20);
    pnh.param("drop_z", drop_z_, -0.83);

    // più alto pre-place, più facile mollare
    pnh.param("place_pre_up", place_pre_up_, 0.28);
    pnh.param("safe_up_after_open", safe_up_after_open_, 0.06); // piccolo shake su

    pnh.param("ik_max_jump", ik_max_jump_, 2.5);
    pnh.param("min_xy_radius", min_xy_radius_, 0.12);

    pnh.param("fixed_roll",  fixed_roll_,  M_PI);
    pnh.param("fixed_pitch", fixed_pitch_, 0.0);

    pnh.param("hand_open",  hand_open_,  0.85);
    pnh.param("hand_close", hand_close_, 0.0);

    //  più tempo di contatto prima di alzare
    pnh.param("grasp_settle_s", grasp_settle_s_, 0.55);

    //  come vuoi tu: NON tornare home dopo place
    pnh.param("use_home_after_place", use_home_after_place_, false);

    loadHomeJoints(pnh);

    sub_js_       = nh.subscribe(js_topic_, 1, &PickPlaceIK::jsCb, this);
    sub_ack_      = nh.subscribe(ack_topic_, 1, &PickPlaceIK::ackCb, this);
    sub_obj_pose_ = nh.subscribe(obj_pose_topic_, 1, &PickPlaceIK::objPoseCb, this);
    sub_obj_rpy_  = nh.subscribe(obj_rpy_topic_,  1, &PickPlaceIK::objRpyCb,  this);

    pub_target_ = nh.advertise<std_msgs::Float64MultiArray>(target_topic_, 1);

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

  void spin() {
    ros::Rate r(50);
    while (ros::ok()) {
      ros::spinOnce();

      geometry_msgs::PoseStamped obj;
      std::vector<double> q_seed8_raw;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!ik_ready_ || !have_js_ || !have_obj_) { r.sleep(); continue; }
        obj = obj_pose_base_;
        q_seed8_raw = q_cur_raw_;
        have_obj_ = false;
      }

      doPickPlace(obj, q_seed8_raw);
      r.sleep();
    }
  }

private:
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


  //  FIX VERO: salto calcolato sul dominio RAW vicino al seed RAW
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
    const double w2 = wrapPi(q6_raw[4]);
    return (std::abs(std::sin(w2)) < 0.15);
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

  void objNameCb(const std_msgs::String& msg) {
  std::lock_guard<std::mutex> lk(mtx_);
  last_obj_name_ = msg.data;
}	

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


  void ackCb(const std_msgs::Bool& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    ack_ = msg.data;
  }

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
      // target “piccolo” -> lo porto vicino al raw attuale
      const double q_small = wrapPi(q_arm6_raw[i]);
      msg.data[i] = wrapNear(q_small, qref8_raw[i]);
    }

    msg.data[6] = q_gripper_cmd_[0];
    msg.data[7] = q_gripper_cmd_[1];
    pub_target_.publish(msg);
  }

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

      lim_out["shoulder_lift_joint"] = {-3.14, 3.14};
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
    for (unsigned int j = 0; j < nj; ++j) { q_min_(j) = -2.0*M_PI; q_max_(j) = 2.0*M_PI; }

    std::map<std::string, std::pair<double,double>> lim;
    const bool ok_yaml = loadJointLimitsFromYaml(joint_limits_yaml_, lim);

    fk_.reset(new KDL::ChainFkSolverPos_recursive(chain_));
    ik_vel_.reset(new KDL::ChainIkSolverVel_pinv(chain_));
    ik_pos_.reset(new KDL::ChainIkSolverPos_NR_JL(chain_, q_min_, q_max_, *fk_, *ik_vel_, 200, 1e-5));
    ik_pos_nolimit_.reset(new KDL::ChainIkSolverPos_NR(chain_, *fk_, *ik_vel_, 200, 1e-5));
    return true;
  }

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

  //  SOLO yaw0 e yaw0+pi => presa “stessa”, niente 90°
  bool solveBestYaw(double px, double py, double pz,
                    double yaw0,
                    const std::vector<double>& seed8_raw,
                    std::vector<double>& qbest_raw,
                    double& best_jump) {

    const double cands[2] = {
      wrapPi(yaw0),
      wrapPi(yaw0 + M_PI)
    };

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
void doPickPlace(const geometry_msgs::PoseStamped& obj, std::vector<double> q_seed8_raw)
{
  const double x = obj.pose.position.x;
  const double y = obj.pose.position.y;
  const double z = obj.pose.position.z;

  // -------------------------
  // Safety base radius
  // -------------------------
  const double rxy = std::sqrt(x*x + y*y);
  if (rxy < min_xy_radius_) {
    ROS_WARN("Object too close to base: r=%.3f < %.3f. Skipping.", rxy, min_xy_radius_);
    return;
  }

  // -------------------------
  // Z safety (tavolo)
  // -------------------------
  const double z_min  = table_z_ + z_clear_;
  const double z_safe = std::max(z, z_min);
  const double drop_z_safe = std::max(drop_z_, z_min);

  // -------------------------
  // Leggo nome oggetto (classe)
  // -------------------------
  std::string obj_name;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    obj_name = last_obj_name_;
  }

  int Xc=0, Yc=0, Zc=0;
  int class_id = -1;
  if (parseXYZ(obj_name, Xc, Yc, Zc)) {
    class_id = classIdFromXYZ(Xc, Yc, Zc);
  }
  
  // --- quote di avvicinamento (default)
double z_pre_high = z_safe + z_pre_off_ + 0.10;
double z_pre      = z_safe + z_pre_off_;
double z_grasp    = z_safe + z_grasp_off_;

//  FIX: se è 2x2 (classe 6) NON schiacciare sul tavolo
// alza la quota grasp di 8-12mm
if (class_id == 6) {
  z_grasp += 0.014;     // +10 mm (puoi provare 0.008 / 0.012)
  z_pre   += 0.014;
}
  
  
  //  close diverso solo per 2x2
double close_cmd = hand_close_;
if (class_id == 6) close_cmd = -0.08;

  // -------------------------
  // Drop Y per classe (modifica qui se vuoi)
  // -------------------------
  auto dropYForClass = [&](int cid)->double {
    switch (cid) {
      case 6: return  0.40;
      case 5: return  0.32;
      case 2: return  0.24;
      case 3: return  0.16;
      case 4: return  0.07;
      case 1: return  0.0;
      case 0: return -0.07;
      default: return drop_y_;   // fallback
    }
  };
  

  //  come vuoi tu: X del place sempre uguale
  const double drop_x_eff = drop_x_;
  double drop_y_eff = dropYForClass(class_id);

  // check rxy drop (evita posizioni impossibili)
  const double r_drop = std::sqrt(drop_x_eff*drop_x_eff + drop_y_eff*drop_y_eff);
  if (r_drop < 0.20 || r_drop > 0.65) {
    ROS_WARN("Drop rxy unsafe (r=%.3f). Forcing safe drop_y=%.3f", r_drop, drop_y_);
    drop_y_eff = drop_y_;
  }

  // -------------------------
  // yaw oggetto
  // -------------------------
  double rr, pp, yaw_obj;
  {
    tf2::Quaternion qtmp;
    tf2::fromMsg(obj.pose.orientation, qtmp);
    tf2::Matrix3x3(qtmp).getRPY(rr, pp, yaw_obj);
  }

  // grasp yaw (come prima)
  const double yaw_grasp = wrapPi(yaw_obj + M_PI/2.0);

  // ✅ place yaw fisso dritto
  const double yaw_place = 0.0;

  ROS_WARN("OBJ base_link x=%.3f y=%.3f z=%.3f (z_safe=%.3f) yaw_obj=%.3f class=%d name=%s -> DROP x=%.3f y=%.3f",
           x, y, z, z_safe, yaw_obj, class_id, obj_name.c_str(), drop_x_eff, drop_y_eff);

  // ------------------------------------------------------------
  // Helper: moveTo BEST yaw (yaw oppure yaw+pi)  --> usato per pick/carry
  // ------------------------------------------------------------
  auto moveToBestYaw = [&](double px, double py, double pz, double yaw,
                           const char* tag, double max_jump)->bool
  {
    std::vector<double> q6_raw;
    double bestJ = 0.0;

    if (!solveBestYaw(px, py, pz, yaw, q_seed8_raw, q6_raw, bestJ)) {
      ROS_WARN("IK failed %s.", tag);
      return false;
    }

    if (bestJ > max_jump) {
      ROS_WARN("IK jump too large at %s: %.3f > %.3f (reject).", tag, bestJ, max_jump);
      return false;
    }

    ROS_INFO("Move %s (jump=%.3f)", tag, bestJ);
    publishJointTarget(q6_raw);

    if (!waitAck(ack_timeout_)) {
      ROS_WARN("Ack timeout %s.", tag);
      return false;
    }

    q_seed8_raw = makeSeed8(q6_raw, q_gripper_cmd_);
    return true;
  };

  // ------------------------------------------------------------
  // Helper: moveTo FIXED yaw (solo yaw richiesto) --> usato per place yaw=0
  // ------------------------------------------------------------
  auto moveToFixedYaw = [&](double px, double py, double pz, double yaw,
                            const char* tag, double max_jump)->bool
  {
    std::vector<double> q6_raw;

    geometry_msgs::Pose p = makeTopDownPose(px, py, pz, yaw);
    if (!solveIK(p, q_seed8_raw, q6_raw)) {
      ROS_WARN("IK failed %s (fixed yaw).", tag);
      return false;
    }

    if (nearWristSingularity(q6_raw)) {
      ROS_WARN("Near singularity at %s (fixed yaw).", tag);
      return false;
    }

    const double j = jumpNorm6(q6_raw, q_seed8_raw);
    if (j > max_jump) {
      ROS_WARN("IK jump too large at %s: %.3f > %.3f (reject).", tag, j, max_jump);
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
  };

  // ------------------------------------------------------------
  // OPEN PRE
  // ------------------------------------------------------------
  ROS_INFO("Opening gripper (pre) (g=%.3f)", hand_open_);
  q_gripper_cmd_[0] = hand_open_;
  q_gripper_cmd_[1] = hand_open_;
  {
    std::vector<double> q6_raw(6);
    for (int i = 0; i < 6; ++i) q6_raw[i] = q_seed8_raw[i];
    publishJointTarget(q6_raw);
    waitAck(ack_timeout_);
    ros::Duration(0.05).sleep();
  }

  // ------------------------------------------------------------
  // APPROACH PICK (anti-sing per X negativa)
  // ------------------------------------------------------------
 

  const double x_front = 0.10;
  const double x_mid0  = 0.00;
  const double x_midN  = -0.12;

  if (x < 0.05) {
    if (!moveToBestYaw(x_front, y, z_pre_high, yaw_grasp, "pre_approach_front", 4.5)) return;
    if (!moveToBestYaw(x_mid0,  y, z_pre_high, yaw_grasp, "pre_step_center",    4.5)) return;
    if (!moveToBestYaw(x_midN,  y, z_pre_high, yaw_grasp, "pre_step_left",      4.5)) return;
  }

  if (!moveToBestYaw(x, y, z_pre_high, yaw_grasp, "pregrasp_high", 4.5)) return;
  if (!moveToBestYaw(x, y, z_pre,      yaw_grasp, "pregrasp",      ik_max_jump_)) return;
  if (!moveToBestYaw(x, y, z_grasp,    yaw_grasp, "grasp",         ik_max_jump_)) return;

  // ------------------------------------------------------------
  // CLOSE
  // ------------------------------------------------------------
  ROS_INFO("Closing gripper (hold) (g=%.3f)", close_cmd);
q_gripper_cmd_[0] = close_cmd;
q_gripper_cmd_[1] = close_cmd;

{
  std::vector<double> q6_raw(6);
  for (int i = 0; i < 6; ++i) q6_raw[i] = q_seed8_raw[i];
  publishJointTarget(q6_raw);
  waitAck(ack_timeout_);
  ros::Duration(grasp_settle_s_).sleep();
}
  // ------------------------------------------------------------
  // MICRO LIFT + LIFT
  // ------------------------------------------------------------
  const double z_micro = z_safe + 0.08;
  if (!moveToBestYaw(x, y, z_micro,              yaw_grasp, "micro_lift", ik_max_jump_)) return;
  if (!moveToBestYaw(x, y, z_safe + z_lift_off_, yaw_grasp, "lift",       ik_max_jump_)) return;

  // ------------------------------------------------------------
  // CARRY UP
  // ------------------------------------------------------------
  const double z_carry = z_safe + 0.30;
  if (!moveToBestYaw(x, y, z_carry, yaw_grasp, "carry_up", ik_max_jump_)) return;

  // waypoint intermedio
  const double x_mid = 0.25;
  moveToBestYaw(x_mid, y, z_carry, yaw_grasp, "carry_mid", 4.5);

  // ------------------------------------------------------------
  // SAFE LANE (evita crash tra Y positive/negative)
  // ------------------------------------------------------------
  const double y_lane = 0.20;   // corsia stabile

  // vai sopra drop mantenendo SEMPRE la stessa X del place
  if (!moveToBestYaw(drop_x_eff, y_lane,    z_carry, yaw_grasp, "carry_lane_safeY",     4.5)) return;
  if (!moveToBestYaw(drop_x_eff, drop_y_eff, z_carry, yaw_grasp, "carry_lane_to_dropY", 4.5)) return;

  // allinea yaw=0 restando sopra il drop
  if (!moveToFixedYaw(drop_x_eff, drop_y_eff, z_carry, yaw_place, "yaw_align_place", 4.5)) {
    ROS_WARN("yaw_align_place failed -> trying yaw=pi fallback");
    if (!moveToFixedYaw(drop_x_eff, drop_y_eff, z_carry, wrapPi(yaw_place + M_PI), "yaw_align_place_pi", 4.5)) return;
  }

  // ------------------------------------------------------------
  // PLACE: scendo davvero quasi a contatto (così LO MOLLA)
  // ------------------------------------------------------------
  if (!moveToFixedYaw(drop_x_eff, drop_y_eff, drop_z_safe + place_pre_up_, yaw_place, "place_pre", 4.5)) return;

  // touch quasi tavolo
  const double z_touch = drop_z_safe + 0.006;
  if (!moveToFixedYaw(drop_x_eff, drop_y_eff, z_touch, yaw_place, "place_touch", 4.5)) return;

  ros::Duration(0.12).sleep();  // settle

  // ------------------------------------------------------------
  // OPEN RELEASE
  // ------------------------------------------------------------
  ROS_INFO("Opening gripper (release) (g=%.3f)", hand_open_);
  q_gripper_cmd_[0] = hand_open_;
  q_gripper_cmd_[1] = hand_open_;
  {
    std::vector<double> q6_raw(6);
    for (int i = 0; i < 6; ++i) q6_raw[i] = q_seed8_raw[i];
    publishJointTarget(q6_raw);
    waitAck(ack_timeout_);
    waitGripperAt(hand_open_, 0.02, 1.0);
    ros::Duration(0.05).sleep();
  }

  // micro peel-off: stacca il blocco dalle dita
  moveToFixedYaw(drop_x_eff, drop_y_eff, drop_z_safe + 0.030, yaw_place, "release_up_small", 4.5);
  moveToFixedYaw(drop_x_eff, drop_y_eff, drop_z_safe + 0.120, yaw_place, "detach_up",        4.5);

  // ------------------------------------------------------------
  // RETREAT su corsia (così il prossimo pick non gira al contrario)
  // ------------------------------------------------------------
  moveToFixedYaw(drop_x_eff, y_lane, z_carry, yaw_place, "retreat_lane", 4.5);

  ROS_INFO("Pick&place done. class=%d name=%s drop=(%.3f,%.3f) placeYaw=0",
           class_id, obj_name.c_str(), drop_x_eff, drop_y_eff);
}
  // ---------- Members ----------
  std::mutex mtx_;

  std::string obj_name_topic_;
  ros::Subscriber sub_obj_name_;
  std::string last_obj_name_;

  std::string joint_limits_yaml_;
  double min_xy_radius_ = 0.15;
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
