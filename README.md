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
<p>First run all driver nodes of the sensors (IMU, 3D LiDAR, 2D LiDAR, camera etc) of your own systems, then run the interfaces nodes to the robot chassis and gimbal motor. Then launch this active gaze control node by calling `ros2 launch livox_gaze_control gaze_control_launch.py` </p>


