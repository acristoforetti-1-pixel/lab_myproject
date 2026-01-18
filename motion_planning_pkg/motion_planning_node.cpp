#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Bool.h>

#include <mutex>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

class MotionPlanningNode {
public:
  MotionPlanningNode()
  : have_map_(false),
    have_js_(false),
    executing_(false),
    last_cmd_valid_(false)
  {
    ros::NodeHandle nh, pnh("~");

    // due tempi separati: braccio più lento, dita più rapide
    pnh.param("arm_T",  arm_T_,  1.0);
    pnh.param("hand_T", hand_T_, 0.25);

    pnh.param<std::string>("joint_states_topic", js_topic_, "/ur5/joint_states");
    pnh.param<std::string>("target_topic",       target_topic_, "/ur5/joint_target");
    pnh.param<std::string>("command_topic",      cmd_topic_, "/ur5/joint_group_pos_controller/command");
    pnh.param<std::string>("ack_topic",          ack_topic_, "/acknowledgement");

    pnh.param("hand_min", hand_min_, 0.0);
    pnh.param("hand_max", hand_max_, 0.8);

    pnh.param("hold_last_command", hold_last_command_, true);
    pnh.param("hold_rate_hz", hold_rate_hz_, 30.0);

    sub_js_     = nh.subscribe(js_topic_, 1, &MotionPlanningNode::jsCb, this);
    sub_target_ = nh.subscribe(target_topic_, 1, &MotionPlanningNode::targetCb, this);

    pub_cmd_ = nh.advertise<std_msgs::Float64MultiArray>(cmd_topic_, 1);
    pub_ack_ = nh.advertise<std_msgs::Bool>(ack_topic_, 1, true); // latched

    q_.assign(6, 0.0);
    q0_.assign(6, 0.0);
    qf_.assign(6, 0.0);

    hand_.assign(2, 0.0);
    hand0_.assign(2, 0.0);
    handf_.assign(2, 0.0);

    last_cmd_.assign(8, 0.0);

    publishAck(false);

    ROS_INFO("MotionPlanningNode:");
    ROS_INFO("  js:  %s", js_topic_.c_str());
    ROS_INFO("  tgt: %s", target_topic_.c_str());
    ROS_INFO("  cmd: %s", cmd_topic_.c_str());
    ROS_INFO("  ack: %s", ack_topic_.c_str());
    ROS_INFO("  arm_T=%.3f  hand_T=%.3f", arm_T_, hand_T_);
    ROS_INFO("  hand clamp: [%.3f, %.3f]", hand_min_, hand_max_);
  }

  void spin() {
    ros::Rate rate(200);
    ros::Time last_hold_pub = ros::Time(0);

    while (ros::ok()) {
      ros::spinOnce();

      if (executing_) {
        step();
      } else if (hold_last_command_ && last_cmd_valid_) {
        const double dt = (ros::Time::now() - last_hold_pub).toSec();
        if (dt > (1.0 / std::max(1.0, hold_rate_hz_))) {
          publishCmdRaw(last_cmd_);
          last_hold_pub = ros::Time::now();
        }
      }

      rate.sleep();
    }
  }

private:
  static double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
  }
  static double clamp01(double x) {
    return clamp(x, 0.0, 1.0);
  }

  static double wrapPi(double a) {
    while (a >  M_PI) a -= 2.0*M_PI;
    while (a < -M_PI) a += 2.0*M_PI;
    return a;
  }

  // differenza angolare modulo 2pi
  static double angDiff(double a, double b) {
    return wrapPi(a - b);
  }

  void jsCb(const sensor_msgs::JointState& msg) {
    if (msg.name.size() != msg.position.size()) return;

    static const std::vector<std::string> arm_names = {
      "shoulder_pan_joint",
      "shoulder_lift_joint",
      "elbow_joint",
      "wrist_1_joint",
      "wrist_2_joint",
      "wrist_3_joint"
    };
    static const std::vector<std::string> hand_names = {
      "hand_1_joint",
      "hand_2_joint"
    };

    std::lock_guard<std::mutex> lk(mtx_);

    if (!have_map_) {
      idx_arm_.assign(6, -1);
      idx_hand_.assign(2, -1);

      for (int i = 0; i < (int)msg.name.size(); ++i) {
        for (int k = 0; k < 6; ++k)
          if (msg.name[i] == arm_names[k]) idx_arm_[k] = i;
        for (int k = 0; k < 2; ++k)
          if (msg.name[i] == hand_names[k]) idx_hand_[k] = i;
      }

      bool ok_arm = true;
      for (int k = 0; k < 6; ++k) ok_arm = ok_arm && (idx_arm_[k] >= 0);
      if (!ok_arm) {
        ROS_ERROR("Some UR5 arm joints not found in joint_states.");
        return;
      }

      bool ok_hand = (idx_hand_[0] >= 0 && idx_hand_[1] >= 0);
      if (!ok_hand) {
        ROS_WARN("hand_1_joint/hand_2_joint not found -> will publish clamped.");
      }

      have_map_ = true;
      ROS_INFO("Joint name->index map created.");
    }

    // ✅ IMPORTANTE: qui NON wrappiamo i joint del braccio!
    // li lasciamo RAW/unwrapped perché il controller vive così.
    for (int k = 0; k < 6; ++k) q_[k] = msg.position[idx_arm_[k]];

    // gripper clampato (questo controller di solito lavora 0..0.8)
    if (idx_hand_[0] >= 0 && idx_hand_[1] >= 0) {
      hand_[0] = clamp(msg.position[idx_hand_[0]], hand_min_, hand_max_);
      hand_[1] = clamp(msg.position[idx_hand_[1]], hand_min_, hand_max_);
    } else {
      hand_[0] = clamp(hand_[0], hand_min_, hand_max_);
      hand_[1] = clamp(hand_[1], hand_min_, hand_max_);
    }

    have_js_ = true;

    if (!last_cmd_valid_) {
      for (int i = 0; i < 6; ++i) last_cmd_[i] = q_[i];
      last_cmd_[6] = hand_[0];
      last_cmd_[7] = hand_[1];
      last_cmd_valid_ = true;
    }
  }

  void targetCb(const std_msgs::Float64MultiArray& msg) {
    if (msg.data.size() < 6) {
      ROS_WARN("Target size < 6 ignored.");
      return;
    }

    std::lock_guard<std::mutex> lk(mtx_);
    if (!have_js_) {
      ROS_WARN("No joint_states yet, target ignored.");
      return;
    }

    q0_ = q_;
    hand0_ = hand_;

    // target braccio (6) arriva tipicamente in [-pi,pi] dal task node
    for (int i = 0; i < 6; ++i) qf_[i] = msg.data[i];

    // dita
    if (msg.data.size() >= 8) {
      handf_[0] = clamp(msg.data[6], hand_min_, hand_max_);
      handf_[1] = clamp(msg.data[7], hand_min_, hand_max_);
    } else {
      handf_[0] = hand_[0];
      handf_[1] = hand_[1];
    }

    // ✅ CHIAVE: porta qf vicino a q0 MA nel dominio UNWRAPPED del controller
    for (int i = 0; i < 6; ++i) {
      // qf è tipo [-pi,pi], q0 può essere 60 rad -> lo "srotolo" vicino a q0
      qf_[i] = q0_[i] + angDiff(qf_[i], wrapPi(q0_[i]));
    }

    if (executing_) ROS_WARN("New target while executing: restart.");

    publishAck(false);
    t_start_ = ros::Time::now();
    executing_ = true;
  }

  void step() {
    std::vector<double> q0, qf, hand0, handf;
    ros::Time t0;
    double Ta, Th;

    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (!executing_) return;
      q0 = q0_; qf = qf_;
      hand0 = hand0_; handf = handf_;
      t0 = t_start_;
      Ta = arm_T_;
      Th = hand_T_;
    }

    const double t = (ros::Time::now() - t0).toSec();

    // progress braccio e mano separati
    const double tau_a = clamp01(t / std::max(1e-3, Ta));
    const double tau_h = clamp01(t / std::max(1e-3, Th));

    const double sa = 3*tau_a*tau_a - 2*tau_a*tau_a*tau_a;
    const double sh = 3*tau_h*tau_h - 2*tau_h*tau_h*tau_h;

    std::vector<double> qd(6);
    for (int i = 0; i < 6; ++i) qd[i] = q0[i] + sa*(qf[i] - q0[i]);

    std::vector<double> hd(2);
    hd[0] = clamp(hand0[0] + sh*(handf[0] - hand0[0]), hand_min_, hand_max_);
    hd[1] = clamp(hand0[1] + sh*(handf[1] - hand0[1]), hand_min_, hand_max_);

    publishCmd8(qd, hd);

    if (t >= std::max(Ta, Th)) {
      std::lock_guard<std::mutex> lk(mtx_);
      q_ = qf_;
      hand_ = handf_;
      executing_ = false;
      publishAck(true);
    }
  }

  void publishCmd8(const std::vector<double>& arm6, const std::vector<double>& hand2) {
    std_msgs::Float64MultiArray cmd;
    cmd.data.resize(8);

    for (int i = 0; i < 6; ++i) cmd.data[i] = arm6[i];
    cmd.data[6] = clamp(hand2[0], hand_min_, hand_max_);
    cmd.data[7] = clamp(hand2[1], hand_min_, hand_max_);

    {
      std::lock_guard<std::mutex> lk(mtx_);
      last_cmd_ = cmd.data;
      last_cmd_valid_ = true;
    }

    pub_cmd_.publish(cmd);
  }

  void publishCmdRaw(const std::vector<double>& cmd8) {
    if ((int)cmd8.size() != 8) return;
    std_msgs::Float64MultiArray cmd;
    cmd.data = cmd8;
    pub_cmd_.publish(cmd);
  }

  void publishAck(bool ok) {
    std_msgs::Bool a;
    a.data = ok;
    pub_ack_.publish(a);
  }

  ros::Subscriber sub_js_, sub_target_;
  ros::Publisher pub_cmd_, pub_ack_;
  std::mutex mtx_;

  std::vector<double> q_, q0_, qf_;
  std::vector<double> hand_, hand0_, handf_;
  std::vector<int> idx_arm_, idx_hand_;

  bool have_map_;
  bool have_js_;
  bool executing_;

  ros::Time t_start_;
  double arm_T_, hand_T_;

  double hand_min_, hand_max_;
  bool hold_last_command_;
  double hold_rate_hz_;
  std::vector<double> last_cmd_;
  bool last_cmd_valid_;

  std::string js_topic_, target_topic_, cmd_topic_, ack_topic_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "motion_planning_node");
  MotionPlanningNode n;
  n.spin();
  return 0;
}
