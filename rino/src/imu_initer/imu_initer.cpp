//
// Created by Yangsc on 23-10-10.
//

#include "imu_initer/imu_initer.hpp"

bool IMUIniter::AddIMU(const IMU &imu) {
    if (init_success_) {
        return true;
    }

    if (init_imu_deque_.empty()) {
        // 记录初始静止时间
        init_start_time_ = imu.timestamp_;
    }

    // 记入初始化队列
    init_imu_deque_.push_back(imu);

    double init_time = imu.timestamp_ - init_start_time_;  // 初始化经过时间
    if (init_time > init_time_seconds_) {
        // 尝试初始化逻辑
        TryInit();
    }

    // 维持初始化队列长度
    while (init_imu_deque_.size() > init_imu_queue_max_size_) {
        init_imu_deque_.pop_front();
    }

    current_time_ = imu.timestamp_;
    return false;
}

bool IMUIniter::TryInit() {
    if (init_imu_deque_.size() < 10) {
        return false;
    }

    // 计算均值和方差
    Vec3d mean_gyro, mean_acce;
    ComputeMeanAndCovDiag(init_imu_deque_, mean_gyro, cov_gyro_, [](const IMU& imu) { return imu.gyro_; });
    ComputeMeanAndCovDiag(init_imu_deque_, mean_acce, cov_acce_, [this](const IMU& imu) { return imu.acce_; });

    // 以acce均值为方向，取9.8长度为重力
    // std::cout << "mean acce: " << mean_acce.transpose() << std::endl;
    gravity_ = - mean_acce / mean_acce.norm() * gravity_norm_;

    // 重新计算加计的协方差
    ComputeMeanAndCovDiag(init_imu_deque_, mean_acce, cov_acce_,
                                [this](const IMU& imu) { return imu.acce_ + gravity_; });

    // 检查IMU噪声
    if (cov_gyro_.norm() > max_static_gyro_var) {
        std::cout << "\e[31m[Warning]Gyroscope measurement noise is too loud \e[0m" << cov_gyro_.norm() << " > " << max_static_gyro_var << std::endl;
        return false;
    }

    if (cov_acce_.norm() > max_static_acce_var) {
        std::cout << "\e[31m[Warning]Accelerometer measurement noise is too loud \e[0m" << cov_acce_.norm() << " > " << max_static_acce_var << std::endl;
        return false;
    }

    // 估计测量噪声和零偏
    init_bg_ = mean_gyro;
    init_ba_ = mean_acce;

    std::cout << "\033[1;34m========= IMU init success! =========\033[0m" << std::endl;
    std::cout << "\033[1;32minitialization time: \033[0m" << current_time_ - init_start_time_ << std::endl;
    std::cout << "\033[1;32mbg = \033[0m" << init_bg_.transpose() << std::endl;
    std::cout << "\033[1;32mba = \033[0m" << init_ba_.transpose() << std::endl;
    std::cout << "\033[1;32mgyro sq = \033[0m" << cov_gyro_.transpose() << std::endl;
    std::cout << "\033[1;32macce sq = \033[0m" << cov_acce_.transpose() << std::endl;
    std::cout << "\033[1;32mgravity = \033[0m" << gravity_.transpose() << "\033[1;32m, norm: \033[0m" << gravity_.norm() << std::endl;
    init_success_ = true;
    return true;
}