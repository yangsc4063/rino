//
// Created by Yangsc on 23-9-20.
//

#include "loader/measure_sync.hpp"


bool MessageSync::Sync() {

    // 如果radar_buffer_或imu_buffer_为空
    if (radar_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

    // 如果radar数据尚未被处理
    if (!radar_pushed_) {
        // 从radar缓冲区中取出数据
        measures_.radar_ = radar_buffer_.front();

        measures_.radar_start_time_ = radar_start_time_;
        measures_.radar_end_time_ = radar_end_time_;
        
        radar_pushed_ = true;
    }

    double imu_time = imu_buffer_.front()->timestamp_;
    measures_.imus_.clear();

    // 循环处理IMU数据，直到时间戳超过radar数据的结束时间
    while ((!imu_buffer_.empty()) && (imu_time < radar_end_time_)) {
        imu_time = imu_buffer_.front()->timestamp_;
        if (imu_time > radar_end_time_) {
            break;
        }
        // 将IMU数据添加到 measures_.imu_ 中
        measures_.imus_.push_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }

    // 弹出 radar_buffer_ 和 time_buffer_ 中的数据
    radar_buffer_.pop_front();
    radar_pushed_ = false;

    if (callback_) {
        callback_(measures_);
    }

    return true;
}