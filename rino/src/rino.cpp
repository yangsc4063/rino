//
// Created by Yangsc on 23-9-20.
//

#include <ros/ros.h>

#include "rino.hpp"


void RINO::IMUCallBack(const sensor_msgs::ImuPtr &msg) {
    sync_->ProcessIMU(msg);
}

void RINO::RadarCallBack(const sensor_msgs::ImagePtr &msg) {    
    sync_->ProcessRadar(msg);
    // std::cout << "\033[1;34m====== Synchronization results ======\033[0m" << std::endl;
    // std::cout << "\033[1;32mtimestamp:\033[0m" << curr_meas_.radar_->timestamp_ << std::endl;
    // std::cout << "\033[1;32mwith radar size: " << 1 << std::endl;
    // std::cout << "\033[1;32mwith imu size: " << curr_meas_.imus_.size() << std::endl;
    // std::cout << std::endl;

    // 可视化关键点提取
    if (viz_extraction_) {
        cv::Mat back_ground;
        // draw_points(curr_meas_.radar_->img_in_cart_, curr_meas_.radar_->kp_, back_ground);
        draw_points(curr_meas_.radar_->img_in_cart_, curr_meas_.radar_->feat_in_cart_, 0.2592, 964, back_ground, {255, 0, 0});
        cv::imshow("Feature Extraction Viz", back_ground);
        cv::waitKey(20);
    }

    // 可视化关键点匹配
    if (flg_first_scan_) {
        flg_first_scan_ = false;
        return;
    }
    if (viz_matching_) {
        cv::Mat prev_back_ground;
        draw_points(prev_matched_.radar_->img_in_cart_, prev_matched_.radar_->kp_, prev_back_ground, {2400, 176, 0});
        cv::Mat curr_back_ground;
        draw_points(curr_matched_.radar_->img_in_cart_, curr_matched_.radar_->kp_, curr_back_ground, {255, 0, 0});
        cv::Mat img_concat;
        cv::hconcat(prev_back_ground, curr_back_ground, img_concat);
        cv::imshow("Feature Matching Viz", img_concat);
        cv::waitKey(20);
    }

    // 可视化点云去畸变
    if (viz_undistortion_) {
        cv::Mat prev_back_ground;
        draw_points(prev_matched_.radar_->img_in_cart_, 
                    prev_matched_.radar_->feat_in_cart_, prev_matched_undistorted_, 
                    0.2592, 964,
                    prev_back_ground,
                    {255, 0, 0}, {0, 0, 255});
        cv::Mat curr_back_ground;
        draw_points(curr_matched_.radar_->img_in_cart_, 
                    curr_matched_.radar_->feat_in_cart_, curr_matched_undistorted_, 
                    0.2592, 964,
                    curr_back_ground,
                    {255, 0, 0}, {0, 0, 255});
        cv::imshow("previous Feature Undistortion Viz", prev_back_ground);
        cv::imshow("current Feature Undistortion Viz", curr_back_ground);
        cv::waitKey(20);
    }
}


void RINO::ProcessMeasurements(const MeasureGroup &meas) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // 对Radar数据进行关键点提取和匹配
    PreprocessRadar(meas);

    auto radar_processing_time = std::chrono::high_resolution_clock::now();

    if (imu_need_init_) {
        // 初始化IMU系统
        TryInitIMU();
        return;
    }

    // 利用IMU数据进行状态预测
    Predict();

    auto imu_prediction_time = std::chrono::high_resolution_clock::now();

    // 对点云去畸变
    Undistort();

    auto undistort_time = std::chrono::high_resolution_clock::now();

    // 配准
    Align();

    auto alignment_time = std::chrono::high_resolution_clock::now();

    // 发布
    PublishOdom();

    auto end_time = std::chrono::high_resolution_clock::now();

    // // 计算各部分处理时长
    // auto radar_process_duration = std::chrono::duration_cast<std::chrono::microseconds>(radar_processing_time - start_time).count();
    // std::cout << "radar_process_duration: " << radar_process_duration << "mus" << std::endl;
    // auto imu_prediction_duration = std::chrono::duration_cast<std::chrono::microseconds>(imu_prediction_time - radar_processing_time).count();
    // std::cout << "imu_prediction_duration: " << imu_prediction_duration << "mus" << std::endl;
    // auto undistort_duration = std::chrono::duration_cast<std::chrono::microseconds>(undistort_time - imu_prediction_time).count();
    // std::cout << "undistort_duration: " << undistort_duration << "mus" << std::endl;
    // auto alignment_duration = std::chrono::duration_cast<std::chrono::microseconds>(alignment_time - undistort_time).count();
    // std::cout << "alignment_duration: " << alignment_duration << "mus" << std::endl;
    // auto publish_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - alignment_time).count();
    // std::cout << "publish_duration: " << publish_duration << "mus" << std::endl;

    // // 存入csv文件中
    // std::ofstream csv_file("/home/yangsc4063/SLAM/Radar_SLAM/RINO_release/src/rino/processing_times.csv", std::ios::app | std::ios::ate);
    // if (csv_file.is_open()) {
    //     if (csv_file.tellp() == 0) csv_file << "radar_process_duration, imu_prediction_duration, undistort_duration, alignment_duration, publish_duration\n";
    //     csv_file << radar_process_duration << ", " << imu_prediction_duration << ", " << undistort_duration << ", " << alignment_duration << ", " << publish_duration << "\n";
    // }
}

void RINO::PreprocessRadar(const MeasureGroup &meas) {
    if (flg_first_scan_) {
        curr_meas_ = radar_processer_.extractFeatures(meas);
        return;
    }

    // extract features
    prev_meas_ = curr_meas_;
    curr_meas_ = radar_processer_.extractFeatures(meas);
    
    // match features
    std::tie(prev_matched_, curr_matched_) = radar_processer_.matchFeatures(prev_meas_, curr_meas_);
}

void RINO::TryInitIMU() {
    for (const auto &imu : curr_meas_.imus_) {
        imu_initer_.AddIMU(*imu);
    }

    if (imu_initer_.InitSuccess()) {
        double gyro_var = sqrt(imu_initer_.GetCovGyro()[0]);
        double acce_var = sqrt(imu_initer_.GetCovAcce()[0]);
        eskf_.SetInitialConditions(gyro_var, acce_var, 
                                    imu_initer_.GetInitBg(), imu_initer_.GetInitBa(), 
                                    imu_initer_.GetGravity());
        imu_need_init_ = false;

        std::cout << "\033[1;32mInitialization parameters have been passed to eskf! \033[0m" << std::endl;
    }
}

void RINO::Predict() {
    if (flg_first_pred_) {
        // 
        curr_timestamp_ = curr_meas_.radar_->timestamp_;

        curr_imu_states_.clear();
        curr_imu_states_.emplace_back(eskf_.GetNominalState());

        // 对IMU状态进行预测
        for (auto &imu : curr_meas_.imus_) {
            eskf_.Predict(*imu);
            curr_imu_states_.emplace_back(eskf_.GetNominalState());
        }
        return;
    }

    prev_imu_states_ = curr_imu_states_;
    
    curr_timestamp_ = curr_meas_.radar_->timestamp_;

    curr_imu_states_.clear();
    curr_imu_states_.emplace_back(eskf_.GetNominalState());

    // 对IMU状态进行预测
    for (auto &imu : curr_meas_.imus_) {
        eskf_.Predict(*imu);
        curr_imu_states_.emplace_back(eskf_.GetNominalState());
    }
}

void RINO::motionDeskewing(const MeasureGroup &meas, const std::vector<NavStated> &imu_states, 
                                  Eigen::Matrix3Xd &feat_undistorted) {
    auto feat = feat_undistorted;
    auto imu_state = imu_states.back();
    SE3 T_end = SE3(imu_state.R_, imu_state.p_);

    for (int i = 0; i < feat.cols(); i++) {
        SE3 Ti;
        // 根据point_times_查找时间，pt.time是该点打到的时间
        PoseInterp<NavStated>(
            meas.radar_->point_times_[i], imu_states, 
            [](const NavStated &s) { return s.timestamp_; },
            [](const NavStated &s) { return s.GetSE3(); },
            Ti);

        Vec3d pi(feat(0, i), feat(1, i), 0);
        Vec3d p_compensate = T_IR_.inverse() * T_end.inverse() * Ti * T_IR_ * pi;

        feat_undistorted(0, i) = p_compensate(0);
        feat_undistorted(1, i) = p_compensate(1);
        feat_undistorted(2, i) = 0;
    }
}

void RINO::dopplerDistortion(const MeasureGroup &meas, const std::vector<NavStated> &imu_states, 
                                    Eigen::Matrix3Xd &feat_undistorted) {
    auto feat = feat_undistorted;
    auto imu_state = imu_states.back();

    for (int i = 0; i < feat.cols(); i++) {
        Vec3d vi;
        // 根据point_times_查找时间，pt.time是该点打到的时间
        VelInterp<NavStated>(
            meas.radar_->point_times_[i], imu_states, 
            [](const NavStated &s) { return s.timestamp_; },
            [](const NavStated &s) { return s.GetVel(); },
            vi);

        Vec3d pi(feat(0, i), feat(1, i), 0);
        Vec3d p_compensate;
        p_compensate(0) = pi(0) + beta_ * vi.norm() * pi(0)*pi(0) / (pi(0)*pi(0) + pi(1)*pi(1));
        p_compensate(1) = pi(1) + beta_ * vi.norm() * pi(0)*pi(1) / (pi(0)*pi(0) + pi(1)*pi(1));

        feat_undistorted(0, i) = p_compensate(0);
        feat_undistorted(1, i) = p_compensate(1);
        feat_undistorted(2, i) = 0;
    }
}

void RINO::Undistort() {
    if (flg_first_pred_) { return; }

    int n_prev = prev_meas_.radar_->feat_in_cart_.cols();
    int n_curr = curr_meas_.radar_->feat_in_cart_.cols();
    prev_feat_undistorted_.resize(3, n_prev);
    curr_feat_undistorted_.resize(3, n_curr);
    prev_feat_undistorted_ = prev_meas_.radar_->feat_in_cart_;
    for (int i = 0; i < n_prev; ++i) {
        prev_feat_undistorted_(1, i) = -prev_feat_undistorted_(1, i);
    }
    curr_feat_undistorted_ = curr_meas_.radar_->feat_in_cart_;
    for (int i = 0; i < n_curr; ++i) {
        curr_feat_undistorted_(1, i) = -curr_feat_undistorted_(1, i);
    }

    int n_matched = curr_matched_.radar_->feat_in_cart_.cols();
    prev_matched_undistorted_.resize(3, n_matched);
    curr_matched_undistorted_.resize(3, n_matched);
    prev_matched_undistorted_ = prev_matched_.radar_->feat_in_cart_;
    for (int i = 0; i < n_matched; ++i) {
        prev_matched_undistorted_(1, i) = -prev_matched_undistorted_(1, i);
    }
    curr_matched_undistorted_ = curr_matched_.radar_->feat_in_cart_;
    for (int i = 0; i < n_matched; ++i) {
        curr_matched_undistorted_(1, i) = -curr_matched_undistorted_(1, i);
    }

    motionDeskewing(prev_meas_, prev_imu_states_, prev_feat_undistorted_);
    motionDeskewing(curr_meas_, curr_imu_states_, curr_feat_undistorted_);
    dopplerDistortion(prev_meas_, prev_imu_states_, prev_feat_undistorted_);
    dopplerDistortion(curr_meas_, curr_imu_states_, curr_feat_undistorted_);

    solver_->reset();
    solver_->setInputSrcFeat(curr_feat_undistorted_);
    solver_->setInputTgtFeat(prev_feat_undistorted_);

    auto curr_imu_rot_ = (curr_imu_states_.back().R_).inverse() * curr_imu_states_.front().R_;
    auto curr_imu_yaw_ = atan2(curr_imu_rot_.matrix()(1, 0), curr_imu_rot_.matrix()(0, 0));

    if (use_motion_deskew_ && abs(curr_imu_yaw_/M_PI*180) > min_for_md_ && abs(curr_imu_yaw_/M_PI*180) < max_for_md_) {
        motionDeskewing(prev_matched_, prev_imu_states_, prev_matched_undistorted_);
        motionDeskewing(curr_matched_, curr_imu_states_, curr_matched_undistorted_);
    }
    if (use_doppler_compensation_) {
        dopplerDistortion(prev_matched_, prev_imu_states_, prev_matched_undistorted_);
        dopplerDistortion(curr_matched_, curr_imu_states_, curr_matched_undistorted_);
    }
}

void RINO::Align() {

    if (flg_first_pred_) {
        // std::cout << "\033[1;34m[Info]curr_timestamp: \033[0m" << curr_timestamp_ << std::endl;
        // std::cout << "\033[1;34m[Info]pose_observe_.rotationMatrix(): \033[0m" << std::endl;
        // std::cout << Mat3d::Identity() << std::endl;
        // std::cout << "\033[1;34m[Info]pose_observe_.translation(): \033[0m" << std::endl; 
        // std::cout << Vec3d::Zero() << std::endl;
        pose_observe_ = SE3(Mat3d::Identity(), Vec3d::Zero());
        // std::ofstream csv_file("/home/yangsc4063/SLAM/Radar_SLAM/RINO_release/src/rino/rio.csv", std::ios::app | std::ios::ate);
        // if (csv_file.is_open()) {
        //     csv_file << curr_timestamp_ << ",";
        //     for (int i = 0; i < 3; ++i) {
        //         for (int j = 0; j < 4; ++j) {
        //             csv_file << pose_observe_.matrix3x4()(i, j) << ",";
        //         }
        //     }
        //     csv_file << std::endl;
        // }

        flg_first_pred_ = false;
        return;
    }

    pose_predict_ = eskf_.GetNominalSE3();
    
    SE3 initial_guess = pose_observe_.inverse() * pose_predict_;
    // std::cout << "\033[1;34m[Info]initial_guess.rotationMatrix(): \033[0m" << std::endl;
    // std::cout << initial_guess.rotationMatrix() << std::endl;
    // std::cout << "\033[1;34m[Info]initial_guess.translation(): \033[0m" << std::endl; 
    // std::cout << initial_guess.translation() << std::endl;
    SE3 output;
    Vec3d uncertainty;
    int inlier;

    // 判断是否为停止状态
    if (curr_matched_undistorted_.cols() > thr_for_stop_motion) {
        std::cout << "\033[1;33m[Warning]Stop motion detected. Pose estimation is skipped for real-time operation!\033[0m" << std::endl;
        output = SE3(Mat3d::Identity(), Vec3d::Zero());
        uncertainty = Vec3d::Zero();
        inlier = 0;
    }
    else {
        solver_->reset();
        solver_->setInputSource(curr_matched_undistorted_);
        solver_->setInputTarget(prev_matched_undistorted_);
        solver_->computeTransformation(initial_guess, output, uncertainty, inlier);
    }
    // std::cout << "\033[1;34m[Info]output.rotationMatrix(): \033[0m" << std::endl;
    // std::cout << output.rotationMatrix() << std::endl;
    // std::cout << "\033[1;34m[Info]output.translation(): \033[0m" << std::endl; 
    // std::cout << output.translation() << std::endl;

    pose_observe_ = pose_observe_ * output;

    eskf_.ObserveSE3(pose_observe_, uncertainty(0), uncertainty(1), uncertainty(2));
    pose_observe_ = eskf_.GetNominalSE3();
    // std::cout << "\033[1;34m[Info]curr_timestamp: \033[0m" << curr_timestamp_ << std::endl;
    // std::cout << "\033[1;34m[Info]pose_observe_.rotationMatrix(): \033[0m" << std::endl;
    // std::cout << pose_observe_.rotationMatrix() << std::endl;
    // std::cout << "\033[1;34m[Info]pose_observe_.translation(): \033[0m" << std::endl; 
    // std::cout << pose_observe_.translation() << std::endl;

    // std::ofstream csv_file("/home/yangsc4063/SLAM/Radar_SLAM/RINO_release/src/rino/rio.csv", std::ios::app | std::ios::ate);
    // if (csv_file.is_open()) {
    //     csv_file << curr_timestamp_ << ",";
    //     for (int i = 0; i < 3; ++i) {
    //         for (int j = 0; j < 4; ++j) {
    //             csv_file << pose_observe_.matrix3x4()(i, j) << ",";
    //         }
    //     }
    //     csv_file << std::endl;
    // }

    // std::ofstream csv_file("/home/yangsc4063/SLAM/Radar_SLAM/RINO_release/src/rino/uncertainty.csv", std::ios::app | std::ios::ate);
    // if (csv_file.is_open()) {
    //     csv_file << curr_timestamp_ << ",";
    //     for (int i = 0; i < 3; ++i) {
    //         csv_file << uncertainty(i) << ",";
    //     }
    //     csv_file << std::endl;
    // }

    // std::ofstream csv_file("/home/yangsc4063/SLAM/Radar_SLAM/RINO_release/src/rino/inliers.csv", std::ios::app | std::ios::ate);
    // if (csv_file.is_open()) {
    //     csv_file << curr_timestamp_ << ",";
    //     csv_file << inlier << ",";
    //     csv_file << std::endl;
    // }
}

void RINO::PublishOdom() {
    std::string odom_frame  = "odom";
    std::string child_frame = "radar_base";
    
    auto time_now  = ros::Time::now();
    auto set_odom_msg = [&](const SE3 &pose, const double yaw) {
        nav_msgs::Odometry odom;
        odom.header.frame_id       = odom_frame;
        odom.child_frame_id        = child_frame;
        odom.header.stamp          = time_now;
        odom.pose.pose.position.x  = pose.translation().x();
        odom.pose.pose.position.y  = pose.translation().y();
        odom.pose.pose.position.z  = pose.translation().z();
        odom.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0.0, 0.0, yaw);
        return odom;
    };

    Eigen::Matrix3d poseRot   = pose_observe_.rotationMatrix();
    Eigen::Vector3d poseEuler = poseRot.eulerAngles(0, 1, 2);
    float           yaw       = float(poseEuler(2));

    PubOdom.publish(set_odom_msg(pose_observe_, yaw)); // last pose
    std::cout << "\033[1;32m[Tips]Complete to publish pose!\033[0m" << std::endl;

    const float constant_z_nonzero = 2.0;

    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudLocal(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudGlobal(new pcl::PointCloud<pcl::PointXYZ>());
    eigen2pcl(curr_feat_undistorted_, *laserCloudLocal, true);
    pcl::transformPointCloud(*laserCloudLocal, *laserCloudGlobal, pose_observe_.matrix().cast<float>());

    for_each(std::execution::par_unseq, laserCloudLocal->points.begin(), laserCloudLocal->points.end(),
            [&](pcl::PointXYZ &pt) {
                pt.z += constant_z_nonzero;
            });

    PubLaserCloudLocal.publish(cloud2msg(*laserCloudLocal, child_frame));
    PubLaserCloudGlobal.publish(cloud2msg(*laserCloudGlobal, odom_frame));

    static tf2_ros::TransformBroadcaster br;
    geometry_msgs::TransformStamped      alias_transform_msg;
    Eigen::Quaterniond                   q(pose_observe_.rotationMatrix());
    alias_transform_msg.header.stamp            = ros::Time::now();
    alias_transform_msg.transform.translation.x = pose_observe_.translation().x();
    alias_transform_msg.transform.translation.y = pose_observe_.translation().y();
    alias_transform_msg.transform.translation.z = pose_observe_.translation().z();
    alias_transform_msg.transform.rotation.x    = q.x();
    alias_transform_msg.transform.rotation.y    = q.y();
    alias_transform_msg.transform.rotation.z    = q.z();
    alias_transform_msg.transform.rotation.w    = q.w();
    alias_transform_msg.header.frame_id         = odom_frame;
    alias_transform_msg.child_frame_id          = child_frame;
    br.sendTransform(alias_transform_msg);

    std::cout << "\033[1;32m[Tips]Complete to publish cloud!\033[0m" << std::endl;
}