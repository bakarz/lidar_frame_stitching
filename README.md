# Doppler ICP Stitching ROS Node

A ROS 2 node for stitching point cloud data using a Doppler Iterative Closest Point (DICP) algorithm, incorporating radial velocity information for enhanced registration.

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Input Data](#input-data)
- [Parameters](#parameters)
- [Published Topics](#published-topics)
- [Usage](#usage)
- [Algorithm Overview](#algorithm-overview)
- [Notes](#notes)

## Overview

The `DopplerICPStitcher` is a ROS 2 node that processes point cloud data from CSV files, aligns them using a Doppler-augmented ICP algorithm, and publishes the stitched point cloud, current pose, and trajectory. It leverages Doppler velocity data to improve alignment accuracy in dynamic environments.

## Dependencies

### System Requirements
- **ROS 2** (Humble)
- **Python 3.1+**

### Python Packages
- `numpy`
- `pandas`
- `open3d`
- `rclpy`
- `sensor_msgs_py`
- `scipy`
- `matplotlib`

Install dependencies using pip:

```bash
pip install numpy pandas open3d rclpy sensor_msgs_py scipy matplotlib
```

### Additional Libraries
- `glob`
- `os`
- `threading`
- `time`

## Input Data

The node expects point cloud data with the following columns:
- `x`, `y`, `z`: Cartesian coordinates (meters)
- `radial_vel` : Doppler radial velocity (m/s)

CSV files should be stored in a directory specified by the `frames_directory` parameter.

## Parameters

The node supports the following configurable parameters:

| Parameter                | Type   | Default Value                                    | Description                                                                 |
|--------------------------|--------|--------------------------------------------------|-----------------------------------------------------------------------------|
| `frames_directory`       | string | `/home/farness/Downloads/.../csv_point_clouds`   | Directory containing point cloud files                                  |
| `velocity_threshold`      | float  | `20`                                             | Max absolute radial velocity for filtering (m/s)                            |
| `downsample_factor`       | int    | `10`                                             | Factor for uniform point cloud downsampling                                |
| `max_iterations`         | int    | `50`                                              | Max iterations for ICP algorithm                                           |
| `icp_tolerance`          | float  | `1e-5`                                           | Convergence tolerance for ICP                                              |
| `publish_rate`           | int    | `10`                                             | Publishing rate (Hz)                                                       |
| `lambda_doppler`         | float  | `0.01`                                           | Weight for Doppler term in ICP cost function                               |
| `frame_dt`               | float  | `0.1`                                            | Time difference between frames (seconds)                                   |
| `t_vl_x`, `t_vl_y`, `t_vl_z` | float | `1.42`, `0.24`, `1.37`                         | Translation vector components for Doppler calculations                     |
| `reject_outliers`        | bool   | `True`                                           | Enable outlier rejection based on Doppler residuals                        |
| `outlier_thresh`         | float  | `2.0`                                            | Threshold for outlier rejection (m/s)                                      |
| `rejection_min_iters`    | int    | `2`                                              | Min iterations before outlier rejection                                    |
| `geometric_min_iters`    | int    | `0`                                              | Min iterations before applying geometric Huber weights                     |
| `doppler_min_iters`      | int    | `2`                                              | Min iterations before applying Doppler Huber weights                       |
| `geometric_k`            | float  | `0.5`                                            | Huber loss parameter for geometric residuals                               |
| `doppler_k`              | float  | `0.2`                                            | Huber loss parameter for Doppler residuals                                 |
| `max_corr_distance`      | float  | `0.3`                                            | Max correspondence distance for ICP (meters)                               |

Set parameters via a ROS 2 parameter file or command-line arguments.

## Published Topics

- **`stitched_cloud`** (`sensor_msgs/PointCloud2`): Stitched point cloud combining all frames.
- **`icp_pose`** (`geometry_msgs/PoseStamped`): Current sensor pose after ICP alignment.
- **`icp_trajectory`** (`geometry_msgs/PoseArray`): Sensor trajectory as a sequence of poses.

## Usage

1. **Prepare Input Data**:
   - Place CSV files with point cloud data in the `frames_directory`.
   - Ensure files include `x`, `y`, `z`, and `v_radial` (or `radial_vel`) columns.

2. **Run the Node**:
   - Source your ROS 2 workspace:
     ```bash
     source /opt/ros/humble/setup.bash
     ```
   - Launch the node:
     ```bash
     ros2 run lidar_frame_stitching stitch_node
     ```


3. **Visualize Outputs**:
   - Use FOXGLOVE-STUDIO to visualize `stitched_cloud`, `icp_pose`, and `icp_trajectory`.
   - Set the fixed frame to `map` in FOXGLOVE-STUDIO.

## Algorithm Overview

1. **Load Frame**: Reads and filters point cloud data based on `velocity_threshold`.
2. **Preprocess Point Cloud**: Downsamples points and estimates normals using Open3D.
3. **Doppler ICP**:
   - Aligns consecutive point clouds using Doppler and geometric residuals.
   - Applies Huber weights for robust estimation and outlier rejection.
   - Solves transformations via least squares optimization.
4. **Stitch Point Clouds**: Merges aligned point clouds into a global map.
5. **Publish Results**: Outputs the stitched point cloud, pose, and trajectory.

## Notes

- CSV files are assumed to be sorted temporally (via `glob.glob` with sorting).
- The Doppler ICP algorithm enhances alignment in dynamic scenes using velocity data.
- The stitched cloud is downsampled if it exceeds 100,000 points to manage memory.
- Detailed logs provide insights into point counts, filtering, and ICP performance.

