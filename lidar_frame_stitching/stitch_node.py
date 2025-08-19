import os
import glob
import numpy as np
import pandas as pd
import open3d as o3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from sensor_msgs_py import point_cloud2
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time
import matplotlib.pyplot as plt
import threading

class DopplerICPStitcher(Node):
    def __init__(self):
        super().__init__('doppler_icp_stitcher')

        self.declare_parameter("frames_directory", "/home/farness/Downloads/carla-sequences/carla-town05-curved-walls/point_clouds/csv_point_clouds")
        self.declare_parameter("velocity_threshold", 20)
        self.declare_parameter("downsample_factor", 10)
        self.declare_parameter("max_iterations", 5)
        self.declare_parameter("icp_tolerance", 1e-5)
        self.declare_parameter("publish_rate", 10) 
        self.declare_parameter("lambda_doppler", 0.01)
        self.declare_parameter("frame_dt", 0.1)
        self.declare_parameter("t_vl_x", 1.419999122619629)
        self.declare_parameter("t_vl_y", 0.23999977111816406)
        self.declare_parameter("t_vl_z", 1.369999885559082)
        self.declare_parameter("reject_outliers", True)
        self.declare_parameter("outlier_thresh", 2.0)
        self.declare_parameter("rejection_min_iters", 2)
        self.declare_parameter("geometric_min_iters", 0)
        self.declare_parameter("doppler_min_iters", 2)
        self.declare_parameter("geometric_k", 0.5)
        self.declare_parameter("doppler_k", 0.2)
        self.declare_parameter("max_corr_distance", 0.3)

        self.frames_dir = self.get_parameter("frames_directory").value
        self.velocity_threshold = self.get_parameter("velocity_threshold").value

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.pointcloud_pub = self.create_publisher(PointCloud2, 'stitched_cloud', qos_profile)
        self.pose_pub = self.create_publisher(PoseStamped, 'icp_pose', qos_profile)
        self.trajectory_pub = self.create_publisher(PoseArray, 'icp_trajectory', qos_profile)

        self.stitched_pts = np.empty((0, 4))
        self.current_pose = np.eye(4)
        self.trajectory = []
        self.frame_files = sorted(glob.glob(os.path.join(self.frames_dir, "*.csv")))
        self.previous_frame = None
        self.current_frame_idx = 0

        publish_rate = self.get_parameter("publish_rate").value
        self.timer = self.create_timer(1.0 / publish_rate, self.process_next_frame)

    def load_frame(self, filename):
        try:
            df = pd.read_csv(filename)
            if 'radial_vel' in df.columns:
                df = df.rename(columns={'radial_vel': 'v_radial'})
            filtered_df = df[np.abs(df['v_radial']) < self.velocity_threshold][['x', 'y', 'z', 'v_radial']]
            self.get_logger().info(
                f"Loaded {filename}: {len(df)} points, {len(filtered_df)} after velocity filtering "
                f"(|v_radial| < {self.velocity_threshold} m/s)"
            )
            return filtered_df.to_numpy()
        except Exception as e:
            self.get_logger().error(f"Error loading {filename}: {str(e)}")
            return None

    def preprocess_point_cloud(self, points, velocities):
        downsample_factor = self.get_parameter("downsample_factor").value
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        self.get_logger().info(f"Original points: {len(points)}")
        pcd_down = pcd.uniform_down_sample(downsample_factor)
        self.get_logger().info(f"Downsampled points: {len(pcd_down.points)}")

        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=10.0, max_nn=30
            )
        )

        down_points = np.asarray(pcd_down.points)
        down_normals = np.asarray(pcd_down.normals)

        if len(down_points) < len(points):
            nbrs = NearestNeighbors(n_neighbors=1).fit(points)
            _, indices = nbrs.kneighbors(down_points)
            down_velocities = velocities[indices.flatten()]
        else:
            down_velocities = velocities

        return down_points, down_normals, down_velocities, pcd_down

    def huber_weights(self, residuals, k):
        abs_r = np.abs(residuals)
        return np.where(abs_r < k, 1.0, k / abs_r)

    def doppler_icp(self, source_frame, target_frame):
        running = [True]

        source_points = source_frame[:, :3]
        source_vel = source_frame[:, 3]
        target_points = target_frame[:, :3]
        target_vel = target_frame[:, 3]

        max_iter = self.get_parameter("max_iterations").value
        tolerance = self.get_parameter("icp_tolerance").value
        lambda_doppler = self.get_parameter("lambda_doppler").value
        dt = self.get_parameter("frame_dt").value
        t_vl = np.array([
            self.get_parameter("t_vl_x").value,
            self.get_parameter("t_vl_y").value,
            self.get_parameter("t_vl_z").value
        ])
        reject_outliers = self.get_parameter("reject_outliers").value
        outlier_thresh = self.get_parameter("outlier_thresh").value
        rejection_min_iters = self.get_parameter("rejection_min_iters").value
        geometric_min_iters = self.get_parameter("geometric_min_iters").value
        doppler_min_iters = self.get_parameter("doppler_min_iters").value
        geometric_k = self.get_parameter("geometric_k").value
        doppler_k = self.get_parameter("doppler_k").value
        max_corr_distance = self.get_parameter("max_corr_distance").value

        src_pts, src_normals, src_vel, _ = self.preprocess_point_cloud(source_points, source_vel)
        tgt_pts, tgt_normals, _, _ = self.preprocess_point_cloud(target_points, target_vel)

        if len(src_normals) == 0 or len(tgt_normals) == 0:
            self.get_logger().warn("No valid normals computed, returning identity transform")
            running[0] = False
            return np.eye(4), np.inf

        transformation = np.eye(4)
        prev_error = np.inf

        for i in range(max_iter):
            transformed = (transformation[:3, :3] @ src_pts.T + transformation[:3, 3:4]).T

            nbrs = NearestNeighbors(n_neighbors=1).fit(tgt_pts)
            distances, indices = nbrs.kneighbors(transformed)
            distances = distances.flatten()
            closest_pts = tgt_pts[indices.flatten()]
            closest_normals = tgt_normals[indices.flatten()]

            diff = transformed - closest_pts
            ptp_dist = np.abs(np.sum(diff * closest_normals, axis=1))

            norms = np.linalg.norm(src_pts, axis=1)
            norms[norms == 0] = 1.0
            d = src_pts / norms[:, np.newaxis]
            u_theta = Rotation.from_matrix(transformation[:3, :3]).as_euler('xyz')
            u_t = transformation[:3, 3]
            cross_u_t_vl = np.cross(u_theta, t_vl)
            predicted = (1 / dt) * (np.sum(d * u_t, axis=1) + np.sum(d * cross_u_t_vl, axis=1))
            r_v = src_vel - predicted

            huber_weights = self.huber_weights(ptp_dist, 0.5)
            total_error = np.mean(huber_weights * (ptp_dist + 0.05 * np.abs(r_v)))
            self.get_logger().info(f"Iteration {i+1}, Total Error: {total_error:.6f}")

            if np.abs(prev_error - total_error) < tolerance:
                self.get_logger().info(f"Converged after {i+1} iterations")
                break
            prev_error = total_error

            inlier_mask = distances < max_corr_distance
            if reject_outliers and (i + 1) >= rejection_min_iters:
                inlier_mask &= np.abs(r_v) < outlier_thresh
            self.get_logger().info(f"Inliers after filtering: {np.sum(inlier_mask)}")
            if np.sum(inlier_mask) < 10:
                self.get_logger().warn("Too few inliers, stopping ICP")
                break

            transformed = transformed[inlier_mask]
            closest_pts = closest_pts[inlier_mask]
            closest_normals = closest_normals[inlier_mask]
            src_pts_filtered = src_pts[inlier_mask]
            src_vel_filtered = src_vel[inlier_mask]
            ptp_dist_filtered = ptp_dist[inlier_mask]
            r_v_filtered = r_v[inlier_mask]
            d_filtered = d[inlier_mask]

            geometric_w = self.huber_weights(ptp_dist_filtered, geometric_k) if i + 1 >= geometric_min_iters else np.ones_like(ptp_dist_filtered)
            doppler_w = self.huber_weights(r_v_filtered, doppler_k) if i + 1 >= doppler_min_iters else np.ones_like(r_v_filtered)

            A_list = []
            b_list = []
            for j in range(len(transformed)):
                p = src_pts_filtered[j]
                n = closest_normals[j]
                diff_j = transformed[j] - closest_pts[j]
                dist = np.dot(diff_j, n)
                row_g = [
                    n[2] * p[1] - n[1] * p[2],
                    n[0] * p[2] - n[2] * p[0],
                    n[1] * p[0] - n[0] * p[1],
                    n[0], n[1], n[2]
                ]
                weight_g = np.sqrt((1 - lambda_doppler) * geometric_w[j])
                A_list.append(np.array(row_g) * weight_g)
                b_list.append(-dist * weight_g)

                d_j = d_filtered[j]
                dx_tvl = np.cross(d_j, t_vl)
                row_d = [
                    -(1 / dt) * dx_tvl[0],
                    -(1 / dt) * dx_tvl[1],
                    -(1 / dt) * dx_tvl[2],
                    -(1 / dt) * d_j[0],
                    -(1 / dt) * d_j[1],
                    -(1 / dt) * d_j[2]
                ]
                weight_d = np.sqrt(lambda_doppler * doppler_w[j])
                A_list.append(np.array(row_d) * weight_d)
                b_list.append(-r_v_filtered[j] * weight_d)

            A = np.vstack(A_list)
            b = np.array(b_list)

            try:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
            except np.linalg.LinAlgError:
                self.get_logger().warn("Linear system solve failed")
                break

            delta_rot = Rotation.from_euler('xyz', x[:3]).as_matrix()
            delta_trans = x[3:]
            delta_transform = np.eye(4)
            delta_transform[:3, :3] = delta_rot
            delta_transform[:3, 3] = delta_trans
            transformation = delta_transform @ transformation

        running[0] = False

        transformed = (transformation[:3, :3] @ src_pts.T + transformation[:3, 3:4]).T
        distances, _ = nbrs.kneighbors(transformed)
        inlier_rmse = np.sqrt(np.mean(distances**2))

        return transformation, inlier_rmse

    def process_next_frame(self):
        if self.current_frame_idx >= len(self.frame_files):
            self.get_logger().info("All frames processed")
            self.timer.cancel()
            return

        frame_data = self.load_frame(self.frame_files[self.current_frame_idx])
        if frame_data is None:
            self.current_frame_idx += 1
            return

        self.get_logger().info(f"Processing frame {self.current_frame_idx + 1}/{len(self.frame_files)}")

        if self.previous_frame is None:
            self.stitched_pts = frame_data
            self.current_pose = np.eye(4)
            self.trajectory.append(self.current_pose.copy())
            self.previous_frame = frame_data
        else:
            start_time = time.time()
            transform, fitness = self.doppler_icp(self.previous_frame, frame_data)
            end_time = time.time()
            stitching_time = end_time - start_time
            self.get_logger().info(f"Time to stitch frame {self.current_frame_idx + 1}: {stitching_time:.2f} seconds")
            self.get_logger().info(f"ICP fitness (inlier RMSE): {fitness:.4f}")

            relative_pose = transform
            self.current_pose = relative_pose @ self.current_pose
            self.trajectory.append(self.current_pose.copy())

            transformed = (self.current_pose[:3, :3] @ frame_data[:, :3].T + self.current_pose[:3, 3:4]).T
            merged_pts = np.hstack((transformed, frame_data[:, 3:]))

            self.stitched_pts = np.vstack((self.stitched_pts, merged_pts))

            if len(self.stitched_pts) > 100000:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.stitched_pts[:, :3])
                pcd = pcd.uniform_down_sample(self.get_parameter("downsample_factor").value)
                down_points = np.asarray(pcd.points)
                self.stitched_pts = np.hstack((down_points, np.zeros((len(down_points), 1))))

            self.previous_frame = frame_data

        self.publish_pointcloud()
        self.publish_current_pose()
        self.publish_trajectory()

        self.current_frame_idx += 1

    def publish_pointcloud(self):
        if len(self.stitched_pts) == 0:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        cloud_data = np.hstack((
            self.stitched_pts[:, :3],
            self.stitched_pts[:, 3].reshape(-1, 1)
        ))

        cloud_msg = point_cloud2.create_cloud(header, fields, cloud_data)
        self.pointcloud_pub.publish(cloud_msg)

    def publish_current_pose(self):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        translation = self.current_pose[:3, 3]
        rotation = Rotation.from_matrix(self.current_pose[:3, :3]).as_quat()

        pose_msg.pose.position.x = translation[0] 
        pose_msg.pose.position.y = translation[1] 
        pose_msg.pose.position.z = translation[2]
        pose_msg.pose.orientation.x = rotation[0]
        pose_msg.pose.orientation.y = rotation[1]
        pose_msg.pose.orientation.z = rotation[2]
        pose_msg.pose.orientation.w = rotation[3]

        self.pose_pub.publish(pose_msg)

    def publish_trajectory(self):
        if not self.trajectory:
            return

        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"

        for pose in self.trajectory:
            pose_msg = Pose()
            translation = pose[:3, 3]
            rotation = Rotation.from_matrix(pose[:3, :3]).as_quat()

            pose_msg.position.x = translation[0]
            pose_msg.position.y = translation[1]
            pose_msg.position.z = translation[2]
            pose_msg.orientation.x = rotation[0]
            pose_msg.orientation.y = rotation[1]
            pose_msg.orientation.z = rotation[2]
            pose_msg.orientation.w = rotation[3]

            pose_array.poses.append(pose_msg)

        self.trajectory_pub.publish(pose_array)

def main(args=None):
    rclpy.init(args=args)
    node = DopplerICPStitcher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
