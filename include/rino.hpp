//
// Created by Yangsc on 23-9-20.
//

#ifndef RINO_HPP
#define RINO_HPP

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include "tf2_ros/transform_broadcaster.h"

#include "common/math_utils.hpp"
#include "common/radar_utils.hpp"
#include "common/nav_state.hpp"
#include "common/eigen_types.hpp"

#include "loader/measure_sync.hpp"
#include "loader/radar_processer.hpp"
#include "imu_initer/imu_initer.hpp"
#include "eskf/eskf.hpp"
#include "odom/pose_solver.hpp"

class RINO {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    RINO() : nh("~") {
        nh.param<std::string>("dataset", dataset_, "mulran");
        nh.param<std::string>("keypoint_extraction", keypoint_extraction_, "cen2019");
        nh.param<double>("voxel_size", voxel_size_, 0.5);
        nh.param<bool>("use_voxelization", use_voxelization_, true);
        nh.param<bool>("viz_extraction", viz_extraction_, false);
        nh.param<bool>("viz_matching", viz_matching_, false);
        std::cout << "\033[1;32mDataset: " << dataset_ << " \033[0m" << std::endl;
        std::cout << "\033[1;32mKeypoint extraction: " << keypoint_extraction_ << " \033[0m" << std::endl;

        nh.param<bool>("use_motion_deskew", use_motion_deskew_, true);
        nh.param<bool>("use_doppler_compensation", use_doppler_compensation_, true);
        nh.param<bool>("viz_undistortion", viz_undistortion_, false);

        // extrinsic parameters
        Mat3d R_IR;
        Vec3d t_IR;
        R_IR << cos(0.9/180*M_PI), -sin(0.9/180*M_PI), 0, sin(0.9/180*M_PI), cos(0.9/180*M_PI), 0, 0, 0, 1;
        t_IR << 1.57, -0.04, 0.0;
        T_IR_ = SE3(R_IR, t_IR);

        // 创建同步模块
        sync_ = std::make_shared<MessageSync>([this](const MeasureGroup &m) { ProcessMeasurements(m); });

        // 设置同步模块参数
        radar_processer_.SetParams(dataset_, 
                                   keypoint_extraction_, 
                                   voxel_size_, 
                                   use_voxelization_);

        // 设置RadarOdom模块参数
        solver_.reset(new Pose_Solver());
        
        // 设置初始pose
        pose_observe_ = SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

        PubOdom = nh.advertise<nav_msgs::Odometry>("ri_odom", 1000);

        PubLaserCloudLocal  = nh.advertise<sensor_msgs::PointCloud2>("cloud_local", 1000, true);
        PubLaserCloudGlobal = nh.advertise<sensor_msgs::PointCloud2>("cloud_global", 1000, true);

        SubImu = nh.subscribe("/imu/data_raw", 1000, &RINO::IMUCallBack, this);
        SubRadar = nh.subscribe("/radar/polar", 1000, &RINO::RadarCallBack, this);
        // SubRadar = nh.subscribe("/Navtech/Polar", 1000, &RINO::RadarCallBack, this);
    };
    ~RINO() = default;

    // Radar回调函数
    void RadarCallBack(const sensor_msgs::ImagePtr &msg);

    // IMU回调函数
    void IMUCallBack(const sensor_msgs::ImuPtr &msg);

private:
    void ProcessMeasurements(const MeasureGroup &meas);

    // 对Radar数据进行关键点提取和匹配
    void PreprocessRadar(const MeasureGroup &meas);

    // 尝试使IMU初始化
    void TryInitIMU();

    // 利用IMU预测状态信息，预测数据会放入imu_states_里
    void Predict();

    // 对prev_matched_和curr_matched_中的点云去畸变
    void Undistort();
    void motionDeskewing(const MeasureGroup &meas, 
                         const std::vector<NavStated> &imu_states, 
                         Eigen::Matrix3Xd &feat_undistorted);

    void dopplerDistortion(const MeasureGroup &meas, 
                           const std::vector<NavStated> &imu_states, 
                           Eigen::Matrix3Xd &feat_undistorted);

    // 执行一次配准和观测
    void Align();

    // 发布位姿和地图
    void PublishOdom();

private:
    SE3 curr_pose_ = SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

private:
    // measure group
    MeasureGroup prev_meas_;
    MeasureGroup curr_meas_;
    // measure group after matching
    MeasureGroup prev_matched_;
    MeasureGroup curr_matched_;
    // Radar timestamps
    int64_t curr_timestamp_;
    // Undistorted features by deskwing and Doppler compensation
    Eigen::Matrix3Xd prev_feat_undistorted_;
    Eigen::Matrix3Xd curr_feat_undistorted_;
    // Undistorted matched-features by deskwing and Doppler compensation
    Eigen::Matrix3Xd prev_matched_undistorted_;
    Eigen::Matrix3Xd curr_matched_undistorted_;

private:
    // ros节点配置
    ros::NodeHandle nh;

    ros::Publisher PubOdom;
    ros::Publisher PubLaserCloudLocal;
    ros::Publisher PubLaserCloudGlobal;

    ros::Subscriber SubRadar;
    ros::Subscriber SubImu;

private:
    // 同步模块
    std::shared_ptr<MessageSync> sync_;

    // IMU初始化器
    IMUIniter imu_initer_;

    // EKF
    ESKFD eskf_;
    std::vector<NavStated> prev_imu_states_; // ESKF预测期间的状态
    std::vector<NavStated> curr_imu_states_;

    // Radar里程计
    boost::shared_ptr<Pose_Solver> solver_;

private:
    // RadarRrocesser参数配置
    RadarProcesser radar_processer_;

    std::string dataset_;
    std::string keypoint_extraction_;

    double voxel_size_;
    bool use_voxelization_ = true;

    bool viz_extraction_ = false;
    bool viz_matching_ = false;
    bool viz_undistortion_ = false;

    // flags
    bool imu_need_init_ = true;
    bool flg_first_scan_ = true;
    bool flg_first_pred_ = true;

    bool use_motion_deskew_ = true;
    bool use_doppler_compensation_ = true;

    // DOPPLER
    double min_for_md_ = 2;
    double max_for_md_ = 9;
    SE3 T_IR_;
    double beta_ = 0.049;

    //
    int thr_for_stop_motion = 600;

    // Pose;
    SE3 pose_predict_;
    SE3 pose_observe_;
};

#endif // RINO_HPP