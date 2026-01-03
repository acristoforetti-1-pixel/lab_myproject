#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Bool.h>
#include <mutex>
#include <algorithm>
#include <vector>
#include <string>


class MotionPlanningNode {
public:
  MotionPlanningNode() : have_map_(false), have_js_(false), executing_(false), T_(3.0) {
    ros::NodeHandle nh, pnh("~");

    pnh.param("duration", T_, 3.0);  // durata traiettoria [s]

    sub_js_ = nh.subscribe("/ur5/joint_states", 1, &MotionPlanningNode::jsCb, this);
    sub_target_ = nh.subscribe("/ur5/joint_target", 1, &MotionPlanningNode::targetCb, this);

    pub_cmd_ = nh.advertise<std_msgs::Float64MultiArray>(
        "/ur5/joint_group_pos_controller/command", 1);

    pub_ack_ = nh.advertise<std_msgs::Bool>("/acknowledgement", 1);

    q_.assign(6, 0.0);
    q0_.assign(6, 0.0);
    qf_.assign(6, 0.0);
  }

  void spin() {
    ros::Rate rate(1000); // 1 kHz
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

  // Costruisci la mappa name->index una sola volta
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
      ROS_ERROR("Some UR5 joints not found in /ur5/joint_states. Names seen:");
      for (auto& n : msg.name) ROS_ERROR("  %s", n.c_str());
      return;
    }

    have_map_ = true;
  }

  // Riordina q_ nel nostro ordine UR5
  for (int k = 0; k < 6; ++k) {
    q_[k] = msg.position[idx_[k]];
  }

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

    // cubic time scaling
    double tau = t / T_;
	if (tau < 0.0) tau = 0.0;
	if (tau > 1.0) tau = 1.0;
    const double s = 3*tau*tau - 2*tau*tau*tau;

    std::vector<double> qd(6);
    for (int i = 0; i < 6; ++i) {
      qd[i] = q0_[i] + s * (qf_[i] - q0_[i]);
    }

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
  bool have_js_;
  std::vector<int> idx_;
  bool have_map_ = false;
  bool executing_;
  ros::Time t_start_;
  double T_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "motion_planning_node");
  MotionPlanningNode n;
  n.spin();
  return 0;
}

