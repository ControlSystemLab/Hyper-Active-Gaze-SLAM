# Hyper-Active-Gaze-SLAM-simulation-gazebo

A gazebo play ground for demonstarting gaze control with a keyboard-controlled omniderectional simulated robot. With visualization of the feature grid map. Codes are different from the codes used for real robot. Have fun.

## Dependencies

<p>ROS2 (Tested with IRON)
<br>PCL
<br>livox-ros-driver2
<br>gridmap (https://github.com/ANYbotics/grid_map)
<br>gazebo
<br>rm-simulation (https://github.com/LihanChen2004/pb_rm_simulation/tree/main)</p>

## How to use
rm-simulation-package installation is required. After compilation and source, run
```
. /usr/share/gazebo/setup.sh
ros2 launch robot_gazebo main.launch.xml
ros2 launch livox_gaze_control gaze_control_launch.py
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Debug potential issues when initializing robot model 
Could be due to conflict of anacoda, simply `conda deactivate`

## Acknowledgement

Thanks for the works:  

[pb_rm-simulation](https://github.com/LihanChen2004/pb_rm_simulation/tree/main)

[robot mania](https://www.youtube.com/watch?v=76cEpo0pFYU)
