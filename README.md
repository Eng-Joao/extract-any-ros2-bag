# ROS2 Bag Data Extractor

This script includes the `get_bag_data` function, which processes specific topics (CompressedImage, Odometry, NavSatFix, and PointCloud2) from ROS2 bag files (.mcap or .db3). It extracts the data and saves it in accessible formats (.png, .csv, .pcd) within an organized folder structure.

## Requirements:
- OpenCV (https://pypi.org/project/opencv-python/)
- MCAP plugin (https://mcap.dev/guides/getting-started/ros-2)
- Open 3D (https://www.open3d.org/docs/release/getting_started.html)
- ROS2 dependencies (rosbag2_py, rclpy, rosidl_runtime_py)

## Usage:
The topics that are desired to be extracted, can be specified in a .yaml file of the following format:
```yaml
namespace: /robotx  #Optional
topics:
  CompressedImage:
    - /my/topic/compressed
  PointCloud2:
    - /my/topic/point_cloud
  Odometry:
    - /my/topic/odom
  NavSatFix:
    - /my/topic/global
```

The program will automatically search for any .yaml file in the folder where the script is located. 

If a .yaml file exists in the script location, you can run the program as follows:
    ``` python get_bag_data.py {path/to/bag_file} ```

Alternatively, if you want to specify a .yaml file that is not in the script location, use the --topic flag:
    ``` python get_bag_data.py {path/to/bag_file} --topic {path/to/file.yaml} ```

You can also manually specify a topic directly via the command line like this:
    ``` python get_bag_data.py {path/to/bag_file} --topic {/specific/topic} ```


Author: João Campanhã, INESC TEC, MOTUS Robotics
Date: 08/11/2024



