//
// Created by Yangsc on 24-3-19.
//

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <boost/format.hpp>

#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <tf/tf.h>

#include "read_file.hpp"
#include "gt_interp.hpp"


int main(int argc, char *argv[]) {

    ros::init(argc, argv, "gt_generator");
    ros::NodeHandle nh;

    std::string common_dir_ = "/media/yangsc4063/Dataset/Mulran/Sejong01(ROS)";

    std::string global_poses_file_ = common_dir_ + "/global_pose.csv";
    std::string radar_dir_ = common_dir_ + "/sensor_data/radar/polar";
    std::string gt_dir_ = common_dir_ + "/gt";

    // 1.load radar timestamps
    std::vector<std::string> radar_files;
    get_file_names(radar_dir_, radar_files, "png");
    std::vector<int64_t> radar_timestamps;
    int count = 0;
    for (const auto    &radar_file: radar_files) {
        std::vector<std::string> parts_dummy;
        boost::split(parts_dummy, radar_file, boost::is_any_of("."));
        int64_t radar_timestamp = std::stoll(parts_dummy[0]);
        radar_timestamps.push_back(radar_timestamp);

        // std::cout << "\033[1;34m[Info]\033[0m" << count++ << "\033[1;34mth radar timestamps are loaded...\033[0m" << std::endl;
        count++;
    }
    std::cout << std::flush;
    std::cout << "\033[1;34m[Info]Total \033[0m" << count << "\033[1;34m radar timestamps are loaded\033[0m" << std::endl;
    std::sort(radar_timestamps.begin(), radar_timestamps.end());

    // 2.load global poses
    std::vector<Mat4d>   raw_poses;
    std::vector<int64_t> raw_pose_timestamps;
    get_global_poses(global_poses_file_, raw_pose_timestamps, raw_poses);
    std::string raw_poses_pcd_path = gt_dir_ + "/global_pose.pcd";
    std::string raw_poses_csv_path = gt_dir_ + "/global_pose.csv";
    save_pcd_poses(raw_poses_pcd_path, raw_poses);
    save_csv_poses(raw_poses_csv_path, raw_pose_timestamps, raw_poses);
    
    // 3.interpolate poses
    std::vector<Mat4d> sync_poses;
    for (const auto &radar_timestamp: radar_timestamps) {
        Mat4d sync_pose;
        PoseInterp(radar_timestamp, raw_pose_timestamps, raw_poses, sync_pose);
        sync_poses.emplace_back(sync_pose);
    }
    std::string sync_poses_pcd_path = gt_dir_ + "/sync_pose.pcd";
    std::string sync_poses_csv_path = gt_dir_ + "/sync_pose.csv";
    save_pcd_poses(sync_poses_pcd_path, sync_poses);
    save_csv_poses(sync_poses_csv_path, radar_timestamps, sync_poses);

    return 0;
}