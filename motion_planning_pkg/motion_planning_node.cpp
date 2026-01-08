#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Bool.h>

#include <mutex>
#include <vector>
#include <string>
#include <algorithm>

class MotionPlanningNode {
public:
  MotionPlanningNode() : have_map_(false), have_js_(false), executing_(false) {
    ros::NodeHandle nh, pnh("~");

    pnh.param("duration", T_, 3.0);

    pnh.param<std::string>("joint_states_topic", js_topic_, "/ur5/joint_states");
    pnh.param<std::string>("target_topic", target_topic_, "/ur5/joint_target");
    pnh.param<std::string>("command_topic", cmd_topic_, "/ur5/joint_group_pos_controller/command");
    pnh.param<std::string>("ack_topic", ack_topic_, "/acknowledgement");

    sub_js_ = nh.subscribe(js_topic_, 1, &MotionPlanningNode::jsCb, this);
    sub_target_ = nh.subscribe(target_topic_, 1, &MotionPlanningNode::targetCb, this);

    pub_cmd_ = nh.advertise<std_msgs::Float64MultiArray>(cmd_topic_, 1);
    pub_ack_ = nh.advertise<std_msgs::Bool>(ack_topic_, 1);

    q_.assign(6, 0.0);
    q0_.assign(6, 0.0);
    qf_.assign(6, 0.0);

    ROS_INFO("MotionPlanningNode:");
    ROS_INFO("  js:  %s", js_topic_.c_str());
    ROS_INFO("  tgt: %s", target_topic_.c_str());
    ROS_INFO("  cmd: %s", cmd_topic_.c_str());
    ROS_INFO("  ack: %s", ack_topic_.c_str());
  }

  void spin() {
    ros::Rate rate(500);
    while (ros::ok()) {
      ros::spinOnce();
      if (executing_) step();
      rate.sleep();
    }
  }

private:
  void jsCb(const sensor_msgs::JointState& msg) {
    static const std::vector<std::string> ur5_names = {
      "shoulder_pan_joint",
      "shoulder_lift_joint",
      "elbow_joint",
      "wrist_1_joint",
      "wrist_2_joint",
      "wrist_3_joint"
    };

    if (msg.name.size() != msg.position.size()) return;

    std::lock_guard<std::mutex> lk(mtx_);

    if (!have_map_) {
      idx_.assign(6, -1);

      for (int k = 0; k < 6; ++k) {
        for (int i = 0; i < (int)msg.name.size(); ++i) {
          if (msg.name[i] == ur5_names[k]) {
            idx_[k] = i;
            break;
          }
        }
      }

      bool ok = true;
      for (int k = 0; k < 6; ++k) ok = ok && (idx_[k] >= 0);
      if (!ok) {
        ROS_ERROR("Some UR5 joints not found in joint_states. Names seen:");
        for (auto& n : msg.name) ROS_ERROR("  %s", n.c_str());
        return;
      }

      have_map_ = true;
      ROS_INFO("Joint name->index map created.");
    }

    for (int k = 0; k < 6; ++k) q_[k] = msg.position[idx_[k]];
    have_js_ = true;
  }

  void targetCb(const std_msgs::Float64MultiArray& msg) {
    if (msg.data.size() < 6) {
      ROS_WARN("Target size < 6, ignored.");
      return;
    }

    std::lock_guard<std::mutex> lk(mtx_);
    if (!have_js_) {
      ROS_WARN("No joint_states yet, target ignored.");
      return;
    }

    q0_ = q_;
    for (int i = 0; i < 6; ++i) qf_[i] = msg.data[i];

    t_start_ = ros::Time::now();
    executing_ = true;
  }

  static double clamp01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
  }

  void step() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!executing_) return;

    const double t = (ros::Time::now() - t_start_).toSec();
    if (t >= T_) {
      publishCmd(qf_);
      executing_ = false;
      publishAck(true);
      return;
    }

    const double tau = clamp01(t / T_);
    const double s = 3*tau*tau - 2*tau*tau*tau;

    std::vector<double> qd(6);
    for (int i = 0; i < 6; ++i) qd[i] = q0_[i] + s*(qf_[i] - q0_[i]);

    publishCmd(qd);
  }

  void publishCmd(const std::vector<double>& qcmd) {
    std_msgs::Float64MultiArray cmd;
    cmd.data = qcmd;
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
  std::vector<int> idx_;
  bool have_map_;
  bool have_js_;
  bool executing_;
  ros::Time t_start_;
  double T_;

  std::string js_topic_, target_topic_, cmd_topic_, ack_topic_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "motion_planning_node");
  MotionPlanningNode n;
  n.spin();
  return 0;
}

