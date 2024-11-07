//
// Created by Yangsc on 23-9-20.
//

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>

#include <deque>

#include <Eigen/Core>
#include <cv_bridge/cv_bridge.h>

#include "common/measure_utils.hpp"


// 将激光数据和IMU数据同步
class MessageSync {
public:
    using Callback = std::function<void(const MeasureGroup &)>;

    MessageSync(Callback cb) : callback_(cb) {};
    ~MessageSync() = default;

    // 处理IMU数据
    void ProcessIMU(const sensor_msgs::ImuPtr &msg) {
        auto imu = std::make_shared<IMU>();

        imu->timestamp_ = static_cast<int64_t>(msg->header.stamp.toSec() * 1e9);

        Vec3d acce = Vec3d(msg->linear_acceleration.x,
                           msg->linear_acceleration.y,
                           msg->linear_acceleration.z);
        // acce.y() = - acce.y();
        imu->acce_ = acce;

        Vec3d gyro = Vec3d(msg->angular_velocity.x,
                           msg->angular_velocity.y,
                           msg->angular_velocity.z);
        // gyro.x() = -gyro.x();
        // gyro.z() = -gyro.z();
        imu->gyro_ = gyro;

        int64_t timestamp = imu->timestamp_;
        
        if (timestamp < last_timestamp_imu_) {
            std::cout << "\e[31m[Error]Imu loop back, clear buffer\e[0m" << std::endl;
            imu_buffer_.clear();
        }

        last_timestamp_imu_ = timestamp;
        imu_buffer_.emplace_back(imu);
    }

    // 处理Radar数据
    void ProcessRadar(const sensor_msgs::ImagePtr &msg) {
        auto radar = std::make_shared<Radar>();

        radar->timestamp_ = static_cast<int64_t>(msg->header.stamp.toSec() * 1e9);

        // 将sensor_msgs::Image转换为cv::Mat
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
        cv::Mat raw_data;
        cv_ptr->image.convertTo(raw_data, CV_32F);
        rotate(raw_data, raw_data, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::flip(raw_data, raw_data, 0);
        raw_data = raw_data / 255.0;
        radar->raw_data_ = raw_data;

        int64_t timestamp = radar->timestamp_;

        if (last_timestamp_radar_ != 0) {
            radar_start_time_ = last_timestamp_radar_;
            radar_end_time_ = timestamp;
        }
        else {
            radar_end_time_ = timestamp;
        }

        if (timestamp < last_timestamp_radar_) {
            std::cout << "\e[31m[Error]Radar loop back, clear buffer\e[0m" << std::endl;;
            radar_buffer_.clear();
        }

        radar_buffer_.push_back(radar);
        last_timestamp_radar_ = timestamp;

        Sync();
    }

public:
    // Synchronizer参数配置
    MeasureGroup measures_;

private:
    // Synchronizer参数配置
    bool Sync();

    Callback callback_;

    std::deque<RadarPtr> radar_buffer_; // 雷达数据缓冲
    std::deque<IMUPtr>   imu_buffer_;   // imu数据缓冲

    int64_t last_timestamp_imu_ = 0;    // 最近imu时间
    int64_t last_timestamp_radar_ = 0;  // 最近radar时间

    bool radar_pushed_ = false;
    
    int64_t radar_start_time_ = 0;
    int64_t radar_end_time_ = 0;
};