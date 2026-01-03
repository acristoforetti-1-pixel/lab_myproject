#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Bool.h>
#include <vector>

class TaskPlanningNode {
public:
  TaskPlanningNode() : ack_(false) {
    ros::NodeHandle nh;
    pub_target_ = nh.advertise<std_msgs::Float64MultiArray>("/ur5/joint_target", 1);
    sub_ack_ = nh.subscribe("/acknowledgement", 1, &TaskPlanningNode::ackCb, this);
  }

  void run() {
    ros::Rate rate(50);
    auto plan = makePlan();

    for (const auto& q : plan) {
      ack_ = false;
      publishTarget(q);

      ros::Time t0 = ros::Time::now();
      ros::Duration timeout(10.0);

      while (ros::ok() && !ack_ && (ros::Time::now() - t0) < timeout) {
        ros::spinOnce();
        rate.sleep();
      }

      if (!ack_) {
        ROS_WARN("Ack timeout for current target.");
      }
    }
  }

private:
  void ackCb(const std_msgs::Bool& msg) { ack_ = msg.data; }

  void publishTarget(const std::vector<double>& q) {
    std_msgs::Float64MultiArray msg;
    msg.data = q;
    pub_target_.publish(msg);
  }

  std::vector<std::vector<double>> makePlan() {
    std::vector<std::vector<double>> P;

    // Ordine: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
    // Metti qui target realistici per il tuo setup.
    P.push_back({-0.32, -0.78, -2.56, -1.63, -1.57,  3.49});
    P.push_back({-0.12, -0.78, -2.56, -1.63, -1.57,  3.49}); // base +0.2
    P.push_back({-0.32, -0.78, -2.56, -1.63, -1.57,  3.49});

    return P;
  }

  ros::Publisher pub_target_;
  ros::Subscriber sub_ack_;
  bool ack_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "task_planning_node");
  TaskPlanningNode n;
  n.run();
  return 0;
}

