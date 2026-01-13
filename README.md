# Pick and Place
For this project we have used servicies in Locosim. please follow the instuction here to configure the workplace:

https://github.com/idra-lab/locosim.git

clone this repo into ros_ws/src in locosim. This repo is work in progress, some details to run the code will be available soon.

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
-  do everything to launch ur5_generic.py with the sources, exports, roscore before

  motion node:
  - source ~/ros_ws/devel/setup.bash
  - rosrun lab_myproject motion_planning_node
  
  vision node......
  
 spawn blocks:
 - source ~/ros_ws/devel/setup.bash
 - export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/root/ros_ws/src/lab_myproject/models
 -  rosrun lab_myproject spawn_random_blocks.py
   
 task planning:
- source ~/ros_ws/devel/setup.bash
- rosrun lab_myproject task_planning_node \
  _object_pose_topic:=/vision/object_pose \
  _gripper_service:=/move_gripper \
  _drop_x:=0.40 _drop_y:=0.00 _drop_z:=0.05 \
  _gripper_open_diameter:=85.0 _gripper_close_diameter:=20.0
  
 (topic vision depends, drops depend on where you want to drop blocks)


## To Test Vision node

U will need a minimum of 2 terminals:
1. -> terminal 1
2. -> terminal 2

1.  cd ~/ros_ws
    catkin_make (optional)
    source devel/setup.bash
  
2.  cd ~/ros_ws
    source devel/setup.bash

1.  python3 -i src/locosim/robot_control/base_controllers/ur5_generic.py

2.  rosrun lab_myproject spawn_random_blocks.py

2.  rosrun lab_myproject perception_6d_node.py

to see the prediction on RVIZ u will need to press:
Add → Image
Topic: /perception/debug/image_raw
Transport hint: raw
Queue size: 2

3. rostopic list | grep detected_object_pose (if u want to see the classifications of the found objects)