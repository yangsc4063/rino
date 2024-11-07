//
// Created by Yangsc on 23-10-10.
//

#ifndef IMU_INITER_H
#define IMU_INITER_H

#include <iostream>
#include <deque>

#include "common/eigen_types.hpp"
#include "common/math_utils.hpp"
#include "common/imu_utils.hpp"

class IMUIniter {
public:
    // 添加IMU数据
    bool AddIMU(const IMU& imu);

    /// 判定初始化是否成功
    bool InitSuccess() const { return init_success_; }

    // 获取各Cov, bias, gravity
    Vec3d GetCovGyro() const { return cov_gyro_; }
    Vec3d GetCovAcce() const { return cov_acce_; }
    Vec3d GetInitBg() const { return init_bg_; }
    Vec3d GetInitBa() const { return init_ba_; }
    Vec3d GetGravity() const { return gravity_; }

private:
    // 尝试对系统初始化
    bool TryInit();

    double init_time_seconds_ = 5 * 1e9; // 静止时间10s
    int init_imu_queue_max_size_ = 2000;  // 初始化IMU队列最大长度
    double max_static_gyro_var = 0.5;     // 静态下陀螺测量方差
    double max_static_acce_var = 0.05;    // 静态下加计测量方差
    double gravity_norm_ = 9.81;          // 重力大小

    bool init_success_ = false;      // 初始化是否成功
    Vec3d cov_gyro_ = Vec3d::Zero(); // 陀螺测量噪声协方差（初始化时评估）
    Vec3d cov_acce_ = Vec3d::Zero(); // 加计测量噪声协方差（初始化时评估）
    Vec3d init_bg_ = Vec3d::Zero();  // 陀螺初始零偏
    Vec3d init_ba_ = Vec3d::Zero();  // 加计初始零偏
    Vec3d gravity_ = Vec3d::Zero();  // 重力
    bool is_static_ = false;         // 标志车辆是否静止
    std::deque<IMU> init_imu_deque_; // 初始化用的数据
    
    double current_time_ = 0.0;    // 当前时间
    double init_start_time_ = 0.0; // 静止的初始时间
};

#endif //IMU_INITER_H