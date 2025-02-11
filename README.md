# Hyper-Active-Gaze-SLAM

A ROS2-based package for omni-directional robots with gimbal to actively control its gaze, to mitigate feature degradations and harsh movement-induced point cloud distortion, hence can safely navigate through unknown environments.

## Dependencies

<p>ROS2 (Tested with IRON)
<br>PCL
<br>livox-ros-driver2
<br>livox-interfaces (For MID-70)
<br>gridmap (https://github.com/ANYbotics/grid_map)
<br>Atomic SLAM algorithms (Cartographer, Point-LIO, LOAM, Kiss-ICP)</p>

## How to use
First run all driver nodes of the sensors (IMU, 3D LiDAR, 2D LiDAR, camera etc) of your own systems, then run the interfaces nodes to the robot chassis and gimbal motor. Then launch this active gaze control node by calling `ros2 launch livox_gaze_control gaze_control_launch.py`. The node can run and the grid map can display as long as the Livox lidar messgaes are recieved, however, to toggle control of the gaze and update of gridmap, the interfaces to the IMU and gimbal motor as well as odometry and/or TF2 queries from the atomic SLAM algorithms are required. But if you simply want to visualize the gridmap, you can simply initialize the LiVOX driver node then run `ros2 launch livox_gaze_control gaze_control_launch.py`. The display of feature point cloud is not default, you can add it in rviz by finding the topic "edge_point" and "plane_point".

## Debug potential issues when using LiVOX LiDARS (MID 70) with ROS-2
See https://blog.csdn.net/omnas/article/details/145163154

## Acknowledgement

Thanks for the works:  

[LIO-Livox](https://github.com/Olive-Wang/Lio-Livox-noted)
