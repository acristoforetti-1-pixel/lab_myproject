# Pick and Place
For this project we have used servicies in Locosim. please follow the instuction here to configure the workplace:

https://github.com/idra-lab/locosim.git

clone this repo into ros_ws/src in locosim.

## Description
A number of objects (e.g., mega-blocks) are stored without any specific order on a stand
(initial stand) located within the workspace of a robotic manipulator. The manipulator is an
anthropomorphic arm, with a spherical wrist and a two-fingered gripper as end-effector.
The objects can belong to different classes but have a known geometry (coded in the STL files). 
The objective of the project is to use the manipulator to pick the objects in sequence and to position them on 
a different stand according to a specified order (final stand). A calibrated 3D sensor is used to 
locate the different objects and to detect their position in the initial stand. 

## Robot
UR5 with a soft two fingers gripper

## Simulation
physics simulation with Gazebo

## To run

shell 1:
-  do everything to launch ur5_generic.py with the sources, exports
-  source /opt/ros/noetic/setup.bash
-  source ~/ros_ws/devel/setup.bash
-   export ROS_MASTER_URI=http://localhost:11311
-   export ROS_HOSTNAME=localhost

shell 2:
- roscore

shell 3:

 task_planning/motion_planning:
 - rosparam set /robot_description "$(rosparam get /ur5/robot_description)"
- roslaunch lab_myproject pick_place_system.launch

shell 4:

 spawn blocks:
 - export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/root/ros_ws/src/lab_myproject/models
 -  rosrun lab_myproject spawn_random_blocks.py
   
shell 5:

vision node:

- source /root/venv/bin/activate
- source /opt/ros/noetic/setup.bash
- source /root/ros_ws/devel/setup.bash
- rosrun lab_myproject perception_6d_node.py _publish_on_request:=true _yaw_mode:=short _yaw_snap_enable:=true _yaw_snap_step:=1.57079632679 _xy_use_rect_center:=true _xy_center_blend:=0.65

to see the prediction on RVIZ u will need to press:
Add → Image
Topic: /perception/debug/image_raw
Transport hint: raw
Queue size: 2


## Report
Project report: [Download PDF](./Pick_and_Place_report.pdf)

Demo video: https://youtu.be/f1nDkgo8dq8?si=jhaF0ubgvUTynl-L
