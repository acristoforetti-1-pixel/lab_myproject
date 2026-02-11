# Lab MyProject - UR5 Pick & Place

Welcome to the documentation for **Lab MyProject**, a ROS-based pick-and-place system using a **UR5 6-DoF robotic arm** with a soft two-finger gripper.

---

## Project Overview

This project demonstrates autonomous pick-and-place operations in a controlled workspace:

- Objects (e.g., mega-blocks) are placed in random positions on an **initial stand**.
- The **UR5 manipulator** picks each object in sequence and places it on a **final stand** according to a specified order.
- Objects can belong to different classes, with known geometries stored as STL files.
- A calibrated 3D sensor detects object positions and orientations on the initial stand.

The system uses **ROS**, **Gazebo simulation**, and **YOLO-based perception** to perform planning and motion execution.

---

## Robot

- **Manipulator:** UR5 arm (6 DoFs)
- **End-effector:** Soft two-fingered gripper
- **Wrist:** Spherical

---

## Simulation

- Physics simulation is performed using **Gazebo**.
- Objects can be spawned randomly in the simulation for testing.
