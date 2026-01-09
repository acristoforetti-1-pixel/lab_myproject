## Pick and Place
For this project we have used servicies in Locosim. please follow the instuction here to configure the workplace:

https://github.com/idra-lab/locosim.git

clone this repo into ros_ws/src in locosim. This repo is work in progress, some details to run the code will be available soon.

# Description
A number of objects (e.g., mega-blocks) are stored without any specific order on a stand
(initial stand) located within the workspace of a robotic manipulator. The manipulator is an
anthropomorphic arm, with a spherical wrist and a two-fingered gripper as end-effector.
The objects can belong to different classes but have a known geometry (coded in the STL files). 
The objective of the project is to use the manipulator to pick the objects in sequence and to position them on 
a different stand according to a specified order (final stand). A calibrated 3D sensor is used to 
locate the different objects and to detect their position in the initial stand. 

# Robot
UR5 with a soft two fingers gripper

# Simulation
physics simulation with Gazebo
