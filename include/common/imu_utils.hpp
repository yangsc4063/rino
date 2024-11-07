//
// Created by Yangsc on 23-9-20.
//

#ifndef IMU_UTILS_H
#define IMU_UTILS_H

#include <memory>
#include "common/eigen_types.hpp"


// IMU 读数
struct IMU {
    IMU() = default;
    IMU(int64_t t, const Vec3d &gyro, const Vec3d &acce) : timestamp_(t), gyro_(gyro), acce_(acce) {}

    int64_t timestamp_ = 0.0;
    Vec3d gyro_ = Vec3d::Zero();
    Vec3d acce_ = Vec3d::Zero();
};

using IMUPtr = std::shared_ptr<IMU>;

#endif  // IMU_UTILS_H