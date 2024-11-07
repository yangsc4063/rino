//
// Created by Yangsc on 24-3-19.
//

#ifndef READ_FILE_HPP
#define READ_FILE_HPP

#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <boost/algorithm/string.hpp>
#include <algorithm>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include "eigen_types.hpp"


static inline bool exists(const std::string &name) {
    struct stat buffer;
    return !(stat(name.c_str(), &buffer) == 0);
}

// assumes file names are EPOCH times which can be sorted numerically
struct less_than_img {
    inline bool operator()(const std::string &img1, const std::string &img2) {
        std::vector<std::string> parts;
        boost::split(parts, img1, boost::is_any_of("."));
        int64_t i1 = std::stoll(parts[0]);
        boost::split(parts, img2, boost::is_any_of("."));
        int64_t i2 = std::stoll(parts[0]);
        return i1 < i2;
    }
};

std::vector<double> split_line(std::string input, char delimiter, int64_t &timestamp) {
    // The most front part: int64 timestamp
    std::vector<double> answer;
    std::stringstream  ss(input);
    std::string        temp;

    bool check_ts = true;
    while (getline(ss, temp, delimiter)) {
        if (check_ts) {
            timestamp = stoll(temp);
            check_ts  = false;
            continue;
        }
        answer.push_back(stod(temp));
    }
    return answer;
}

void get_file_names(std::string path, std::vector<std::string> &files, std::string extension) {
    DIR           *dirp = opendir(path.c_str());
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {
        if (exists(dp->d_name)) {
            if (!extension.empty()) {
                std::vector<std::string> parts;
                boost::split(parts, dp->d_name, boost::is_any_of("."));
                if (parts[parts.size() - 1].compare(extension) != 0)
                    continue;
            }
            files.push_back(dp->d_name);
        }
    }
    // Sort files in ascending order of time stamp
    std::sort(files.begin(), files.end(), less_than_img());
}

void vec2mat(std::vector<double> &pose_vec, Mat4d &pose_mat) {
    for (int idx = 0; idx < 12; ++idx) {
        int i = (idx) / 4;
        int j = (idx) % 4;
        pose_mat(i, j) = pose_vec[idx];
    }
}

void get_global_poses(std::string global_poses_file, 
                      std::vector<int64_t> &timestamps,
                      std::vector<Mat4d> &poses) {
    poses.clear();
    poses.reserve(4000);
    timestamps.clear();
    timestamps.reserve(4000);

    std::ifstream in(global_poses_file);
    std::string   line;

    int count = 0;
    Mat4d origin = Mat4d::Identity();
    while (getline(in, line)) {
        // Checking delimiter is important!!!
        int64_t timestamp;
        // Timestamp should be parsed separately!
        std::vector<double> parsed_data = split_line(line, ',', timestamp);
        timestamps.push_back(timestamp);

        Mat4d pose = Mat4d::Identity(); // Crucial!
        vec2mat(parsed_data, pose);
        if (count == 0) {
            origin = pose;
        }

        pose = origin.inverse() * pose;
        poses.emplace_back(pose);

        // std::cout << "\033[1;34m[Info]\033[0m" << count++ << "\033[1;34mth pose are loaded...\033[0m" << std::endl;
        count++;
    }
    in.close();
    std::cout << std::flush;
    std::cout << "\033[1;34m[Info]Total \033[0m" << count << "\033[1;34m poses are loaded\033[0m" << std::endl;
}

// Just check the pose by Cloud Compare
void save_pcd_poses(std::string pcd_path, std::vector<Mat4d> &poses) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.points.reserve(poses.size());
    for (const auto &pose: poses) {
        pcl::PointXYZ p(pose(0, 3), pose(1, 3), 0);
        cloud.points.emplace_back(p);
    }
    pcl::io::savePCDFileBinary(pcd_path, cloud);
}

void save_csv_poses(std::string csv_path, std::vector<int64_t> &timestamps, std::vector<Mat4d> &poses) {
    std::ofstream out(csv_path);
    for (int idx = 0; idx < poses.size(); ++idx) {
        out << timestamps[idx] << ",";
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                out << poses[idx](i, j) << ",";
            }
        }
        out << std::endl;
    }
    out.close();
}

#endif // READ_FILE_HPP