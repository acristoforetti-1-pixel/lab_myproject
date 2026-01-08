/*
source ~/ros_ws/devel/setup.bash
rosrun lab_myproject motion_planning_node

source ~/ros_ws/devel/setup.bash
rosrun lab_myproject task_planning_node \
  _ee_link:=ur5::wrist_3_link \
  _expected_objects:=4 \
  _known_models:="['X1-Y1-Z2','X1-Y2-Z1','X1-Y2-Z2','X1-Y2-Z2-CHAMFER','X1-Y2-Z2-TWINFILLET','X1-Y3-Z2','X1-Y3-Z2-FILLET','X1-Y4-Z1','X1-Y4-Z2','X2-Y2-Z2','X2-Y2-Z2-FILLET']"


source ~/ros_ws/devel/setup.bash
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/root/ros_ws/src/lab_myproject/models
rosrun lab_myproject spawn_random_blocks.py


*/
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Bool.h>

#include <gazebo_msgs/GetWorldProperties.h>
#include <gazebo_msgs/GetModelState.h>
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/GetLinkState.h>

#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <mutex>
#include <string>
#include <vector>
#include <algorithm>

class TaskPlannerGazeboFollow {
public:
  TaskPlannerGazeboFollow() : ack_(false), holding_(false) {
    ros::NodeHandle nh, pnh("~");

    // motion interface
    pnh.param<std::string>("target_topic", target_topic_, "/ur5/joint_target");
    pnh.param<std::string>("ack_topic", ack_topic_, "/acknowledgement");

    // gazebo
    pnh.param<std::string>("world_frame", world_frame_, "world");
    pnh.param<std::string>("ee_link", ee_link_, "ur5::wrist_3_link");

    // behavior
    pnh.param("follow_rate_hz", follow_rate_hz_, 120.0);
    pnh.param("ack_timeout", ack_timeout_, 10.0);
    pnh.param("spawn_wait_timeout", spawn_wait_timeout_, 30.0);
    pnh.param("expected_objects", expected_objects_, 4);

    // optional: list of known model prefixes (e.g., ["block","cube","cylinder"])
    pnh.getParam("known_models", known_models_);

    pub_target_ = nh.advertise<std_msgs::Float64MultiArray>(target_topic_, 1);
    sub_ack_ = nh.subscribe(ack_topic_, 1, &TaskPlannerGazeboFollow::ackCb, this);

    cli_world_     = nh.serviceClient<gazebo_msgs::GetWorldProperties>("/gazebo/get_world_properties");
    cli_get_model_ = nh.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
    cli_set_model_ = nh.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
    cli_get_link_  = nh.serviceClient<gazebo_msgs::GetLinkState>("/gazebo/get_link_state");

    // ====== DEMO joint targets ======
    // Metti pose che raggiungono l’area dove spawni i blocchi.
    q_pregrasp_ = {-0.32, -0.78, -2.56, -1.63, -1.57,  3.49};
    q_lift_     = {-0.32, -0.60, -2.30, -1.75, -1.57,  3.49};

    q_drop_[0]  = {-0.60, -0.78, -2.56, -1.63, -1.57, 3.49};
    q_drop_[1]  = {-0.40, -0.78, -2.56, -1.63, -1.57, 3.49};
    q_drop_[2]  = {-0.20, -0.78, -2.56, -1.63, -1.57, 3.49};
    q_drop_[3]  = { 0.00, -0.78, -2.56, -1.63, -1.57, 3.49};

    ROS_INFO("TaskPlannerGazeboFollow READY");
    ROS_INFO("  ee_link=%s world_frame=%s", ee_link_.c_str(), world_frame_.c_str());
    ROS_INFO("  expecting %d objects, wait timeout=%.1fs", expected_objects_, spawn_wait_timeout_);
    if (!known_models_.empty()) {
      ROS_INFO("  known_models prefixes:");
      for (auto &s : known_models_) ROS_INFO("    - %s", s.c_str());
    } else {
      ROS_WARN("  known_models param not set: will pick any model except ground/table/ur5.");
    }
  }

  void run() {
    ros::Rate rate(50);

    // 1) wait for spawned objects
    auto objects = waitForSpawnedObjects(expected_objects_, spawn_wait_timeout_);
    if ((int)objects.size() < expected_objects_) {
      ROS_ERROR("Expected %d objects but got %zu. Aborting.", expected_objects_, objects.size());
      return;
    }

    ROS_INFO("Objects selected:");
    for (auto &o : objects) ROS_INFO("  - %s", o.c_str());

    // 2) pick & place
    for (int i = 0; i < expected_objects_ && i < (int)objects.size(); ++i) {
      const std::string& obj = objects[i];
      ROS_INFO("=== Handling object %d/%d: %s ===", i+1, expected_objects_, obj.c_str());

      sendTarget(q_pregrasp_);
      waitAck(rate, ack_timeout_);

      if (!startHolding(obj)) {
        ROS_WARN("Cannot start holding %s. Skipping.", obj.c_str());
        continue;
      }

      sendTarget(q_lift_);
      waitAck(rate, ack_timeout_);

      sendTarget(q_drop_[i]);
      waitAck(rate, ack_timeout_);

      followForSeconds(0.8);

      stopHolding();
      ROS_INFO("Released %s", obj.c_str());
      ros::Duration(0.2).sleep();
    }

    ROS_INFO("Task complete.");
  }

private:
  // -------- ROS callbacks --------
  void ackCb(const std_msgs::Bool& msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    ack_ = msg.data;
  }

  void sendTarget(const std::vector<double>& q6) {
    std_msgs::Float64MultiArray msg;
    msg.data = q6;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      ack_ = false;
    }
    pub_target_.publish(msg);
  }

  bool waitAck(ros::Rate& rate, double timeout_s) {
    ros::Time t0 = ros::Time::now();
    while (ros::ok()) {
      ros::spinOnce();
      rate.sleep();
      std::lock_guard<std::mutex> lk(mtx_);
      if (ack_) return true;
      if ((ros::Time::now() - t0).toSec() > timeout_s) {
        ROS_WARN("Ack timeout.");
        return false;
      }
    }
    return false;
  }

  // -------- object filtering --------
  static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
  }

  bool isKnownSpawnedObject(const std::string& model_name) const {
    // our spawner creates: "<base>_<8hex>"
    // if known_models is provided, accept only those whose prefix matches any known model.
    if (known_models_.empty()) return true;

    auto lower = toLower(model_name);
    for (const auto& base : known_models_) {
      auto b = toLower(base);
      // match prefix base_   OR base-  OR base (at start)
      if (lower.rfind(b + "_", 0) == 0) return true;
      if (lower.rfind(b + "-", 0) == 0) return true;
      if (lower.rfind(b, 0) == 0) return true;
    }
    return false;
  }

  std::vector<std::string> getCandidateObjectsOnce() {
    gazebo_msgs::GetWorldProperties srv;
    if (!cli_world_.call(srv) || !srv.response.success) return {};

    std::vector<std::string> objects;
    for (auto &m : srv.response.model_names) {
      auto ml = toLower(m);

      // exclude fixed/world stuff
      if (ml.find("ground") != std::string::npos) continue;
      if (ml.find("plane")  != std::string::npos) continue;
      if (ml.find("tavolo") != std::string::npos) continue;
      if (ml.find("table")  != std::string::npos) continue;
      if (ml == "ur5") continue;

      if (!isKnownSpawnedObject(m)) continue;

      objects.push_back(m);
    }
    return objects;
  }

  std::vector<std::string> waitForSpawnedObjects(int n, double timeout_s) {
    ros::Time t0 = ros::Time::now();
    ros::Rate r(5);

    while (ros::ok()) {
      auto objs = getCandidateObjectsOnce();
      if ((int)objs.size() >= n) {
        std::sort(objs.begin(), objs.end());     // stable
        if ((int)objs.size() > n) objs.resize(n); // take first n
        return objs;
      }

      if ((ros::Time::now() - t0).toSec() > timeout_s) {
        ROS_WARN("Spawn wait timeout: found %zu objects.", objs.size());
        return objs;
      }

      ros::spinOnce();
      r.sleep();
    }
    return {};
  }
  
  static geometry_msgs::Pose toPoseMsg(const tf2::Transform& T) {
  geometry_msgs::Pose p;
  p.position.x = T.getOrigin().x();
  p.position.y = T.getOrigin().y();
  p.position.z = T.getOrigin().z();
  p.orientation = tf2::toMsg(T.getRotation());
  return p;
}

  // -------- gazebo pose helpers --------
  bool getModelPoseWorld(const std::string& model, tf2::Transform& w_T_o) {
    gazebo_msgs::GetModelState srv;
    srv.request.model_name = model;
    srv.request.relative_entity_name = world_frame_;
    if (!cli_get_model_.call(srv) || !srv.response.success) return false;
    tf2::fromMsg(srv.response.pose, w_T_o);
    return true;
  }

  bool setModelPoseWorld(const std::string& model, const tf2::Transform& w_T_o) {
    gazebo_msgs::SetModelState srv;
    srv.request.model_state.model_name = model;
    srv.request.model_state.reference_frame = world_frame_;
    srv.request.model_state.pose = toPoseMsg(w_T_o);
    return cli_set_model_.call(srv) && srv.response.success;
  }

  bool getEeLinkPoseWorld(tf2::Transform& w_T_ee) {
    gazebo_msgs::GetLinkState srv;
    srv.request.link_name = ee_link_;
    srv.request.reference_frame = world_frame_;
    if (!cli_get_link_.call(srv) || !srv.response.success) return false;
    tf2::fromMsg(srv.response.link_state.pose, w_T_ee);
    return true;
  }

  // -------- holding logic --------
  bool startHolding(const std::string& model_name) {
    tf2::Transform w_T_ee, w_T_o;
    if (!getEeLinkPoseWorld(w_T_ee)) return false;
    if (!getModelPoseWorld(model_name, w_T_o)) return false;

    ee_T_obj_ = w_T_ee.inverse() * w_T_o;

    holding_model_ = model_name;
    holding_ = true;

    follow_timer_ = nh_.createTimer(ros::Duration(1.0 / follow_rate_hz_),
                                    &TaskPlannerGazeboFollow::followTimerCb, this);
    return true;
  }

  void stopHolding() {
    holding_ = false;
    holding_model_.clear();
    follow_timer_.stop();
  }

  void followTimerCb(const ros::TimerEvent&) {
    if (!holding_) return;

    tf2::Transform w_T_ee;
    if (!getEeLinkPoseWorld(w_T_ee)) return;

    tf2::Transform w_T_o = w_T_ee * ee_T_obj_;
    setModelPoseWorld(holding_model_, w_T_o);
  }

  void followForSeconds(double s) {
    ros::Time t0 = ros::Time::now();
    ros::Rate r(std::max(10.0, follow_rate_hz_));
    while (ros::ok() && (ros::Time::now() - t0).toSec() < s) {
      ros::spinOnce();
      r.sleep();
    }
  }

  // ROS
  ros::NodeHandle nh_;
  ros::Publisher pub_target_;
  ros::Subscriber sub_ack_;
  std::mutex mtx_;
  bool ack_;

  // services
  ros::ServiceClient cli_world_, cli_get_model_, cli_set_model_, cli_get_link_;

  // params
  std::string target_topic_, ack_topic_;
  std::string world_frame_, ee_link_;
  double follow_rate_hz_;
  double ack_timeout_;
  double spawn_wait_timeout_;
  int expected_objects_;
  std::vector<std::string> known_models_;

  // joint plans
  std::vector<double> q_pregrasp_, q_lift_;
  std::vector<double> q_drop_[4];

  // holding
  bool holding_;
  std::string holding_model_;
  tf2::Transform ee_T_obj_;
  ros::Timer follow_timer_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "task_planning_node");
  TaskPlannerGazeboFollow n;
  n.run();
  return 0;
}
