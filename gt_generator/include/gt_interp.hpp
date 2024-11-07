//
// Created by Yangsc on 24-3-19.
//

#ifndef GT_INTERP_HPP
#define GT_INTERP_HPP

#include <vector>
#include <iostream>

#include "eigen_types.hpp"


inline bool PoseInterp(int64_t radar_timestamp, 
                       std::vector<int64_t> pose_timestamps, std::vector<Mat4d> poses,
                       Mat4d& result, float time_th = 0.5) {
    size_t best_match;

    if (pose_timestamps.empty()) {
        std::cout << "\033[1;31m[Warning]Cannot interp because data is empty.\033[0m" << std::endl;
        return false;
    }

    size_t last_idx = pose_timestamps.size() - 1;
    if (radar_timestamp > pose_timestamps[last_idx]) {
        if (radar_timestamp < (pose_timestamps[last_idx] + time_th)) {
            // 尚可接受
            result = poses[last_idx];
            best_match = last_idx;
            return true;
        }
        return false;
    }

    size_t match_idx = 0;
    for (size_t idx = 0; idx < pose_timestamps.size(); ++idx) {
        size_t next_idx = idx + 1;

        if (pose_timestamps[idx] < radar_timestamp && pose_timestamps[next_idx] >= radar_timestamp) {
            match_idx = idx;
            break;
        }
    }

    size_t match_idx_n = match_idx + 1;

    double dt = pose_timestamps[match_idx_n] - pose_timestamps[match_idx];
    double s = (radar_timestamp - pose_timestamps[match_idx]) / dt; // s=0 时为第一帧，s=1时为next
    
    if (fabs(dt) < 1e-6) {
        best_match = match_idx;
        result = poses[match_idx];
        return true;
    }

    Mat4d pose_first = poses[match_idx];
    Mat4d pose_next  = poses[match_idx_n];
    Eigen::Quaterniond q_first(pose_first.block<3, 3>(0, 0));
    // std::cout << "q_first: " << q_first.toRotationMatrix() << std::endl;
    Eigen::Quaterniond q_next(pose_next.block<3, 3>(0, 0));
    // std::cout << "q_next: " << q_next.toRotationMatrix() << std::endl;
    Eigen::Quaterniond q_result = q_first.slerp(s, q_next);
    // std::cout << "q_result: " << q_result.toRotationMatrix() << std::endl;
    Vec3d t_result = pose_first.block<3, 1>(0, 3) * (1 - s) + pose_next.block<3, 1>(0, 3) * s;
    result.block<3, 3>(0, 0) = q_result.toRotationMatrix();
    result.block<3, 1>(0, 3) = t_result;

    best_match = s < 0.5 ? match_idx : match_idx_n;
    return true;
}

#endif //GT_INTERP_HPP