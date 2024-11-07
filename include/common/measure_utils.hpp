//
// Created by Yangsc on 23-9-27.
//

#ifndef MEASURE_UTILS_H
#define MEASURE_UTILS_H

#include <deque>

#include "common/radar_utils.hpp"
#include "common/imu_utils.hpp"


// radar-imu 同步数据
struct MeasureGroup {
    int64_t radar_start_time_ = 0; // radar的起始时间
    int64_t radar_end_time_ = 0;   // radar的终止时间
    RadarPtr radar_;               // radar点云
    std::deque<IMUPtr> imus_;       // 同步时间内IMU读数
};

#endif //MEASURE_UTILS_H