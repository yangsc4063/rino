//
// Created by Yangsc on 23-10-10.
//

#ifndef ESKF_H
#define ESKF_H

#include <iostream>

#include "common/eigen_types.hpp"
#include "common/math_utils.hpp"
#include "common/imu_utils.hpp"
#include "common/nav_state.hpp"


template <typename S = double>
class ESKF {
public:
    using SO3 = Sophus::SO3<S>;                    // 旋转变量类型
    using VecT = Eigen::Matrix<S, 3, 1>;           // 向量类型
    using Vec18T = Eigen::Matrix<S, 18, 1>;        // 18维向量类型
    using Mat3T = Eigen::Matrix<S, 3, 3>;          // 3x3矩阵类型
    using MotionNoiseT = Eigen::Matrix<S, 18, 18>; // 运动噪声类型
    using Mat18T = Eigen::Matrix<S, 18, 18>;       // 18维方差类型
    using NavStateT = NavState<S>;                 // 整体名义状态变量类型

    void SetInitialConditions(const double &gyro_var_, const double &acce_var_,
                              const VecT &init_bg, const VecT &init_ba,
                              const VecT &gravity = VecT(0, 0, -9.8)) {
        BuildNoise(gyro_var_, acce_var_, bias_gyro_var_, bias_acce_var_);
        bg_ = init_bg;
        ba_ = init_ba;
        g_ = gravity;
        cov_ = Mat18T::Identity() * 1e-4;
    }

    // 设置状态X
    void SetX(const NavStated& x, const Vec3d& grav) {
        current_time_ = x.timestamp_;
        R_ = x.R_;
        p_ = x.p_;
        v_ = x.v_;
        bg_ = x.bg_;
        ba_ = x.ba_;
        g_ = grav;
    }

    // 设置协方差
    void SetCov(const Mat18T& cov) { cov_ = cov; }

    // 使用IMU递推
    bool Predict(const IMU &imu);

    /**
     * 使用SE3进行观测
     * @param pose  观测位姿
     * @param trans_noise 平移噪声
     * @param ang_noise   角度噪声
     * @return
     */
    bool ObserveSE3(const SE3& pose, 
                    double ang_noise, double x_noise, double y_noise);

    /// 获取全量状态
    NavStateT GetNominalState() const { return NavStateT(current_time_, R_, p_, v_, bg_, ba_); }

    // 获取SE3状态
    SE3 GetNominalSE3() const { return SE3(R_, p_); }

    // 获取重力
    Vec3d GetGravity() const { return g_; }

private:
    void BuildNoise(const double &gyro_var_, const double &acce_var_,
                    const double &bias_gyro_var_, const double &bias_acce_var_) {
        double et = gyro_var_;
        double ev = acce_var_;
        double eg = bias_gyro_var_;
        double ea = bias_acce_var_;

        double ev2 = ev;  // * ev;
        double et2 = et;  // * et;
        double eg2 = eg;  // * eg;
        double ea2 = ea;  // * ea;

        // 设置过程噪声
        Q_.diagonal() << 0, 0, 0, ev2, ev2, ev2, et2, et2, et2, eg2, eg2, eg2, ea2, ea2, ea2, 0, 0, 0;
    }

    // 更新名义状态变量，重置error state
    void UpdateAndReset() {
        p_ += dx_.template block<3, 1>(0, 0);
        v_ += dx_.template block<3, 1>(3, 0);
        R_ = R_ * SO3::exp(dx_.template block<3, 1>(6, 0));

        if (update_bias_gyro_) {
            bg_ += dx_.template block<3, 1>(9, 0);
        }

        if (update_bias_acce_) {
            ba_ += dx_.template block<3, 1>(12, 0);
        }

        g_ += dx_.template block<3, 1>(15, 0);

        ProjectCov();
        dx_.setZero();
    }

    // 对P阵进行投影，参考式(3.63)
    void ProjectCov() {
        Mat18T J = Mat18T::Identity();
        J.template block<3, 3>(6, 6) = Mat3T::Identity() - 0.5 * SO3::hat(dx_.template block<3, 1>(6, 0));
        cov_ = J * cov_ * J.transpose();
    }

private:
    // IMU 测量与零偏参数
    double imu_dt_ = 0.01;        // IMU测量间隔
    // IMU噪声项都为离散时间，不需要再乘dt，可以由初始化器指定IMU噪声
    double gyro_var_ = 1e-5;      // 陀螺仪测量标准差
    double acce_var_ = 1e-2;      // 加速度计测量标准差
    double bias_gyro_var_ = 1e-6; // 陀螺仪零偏游走标准差
    double bias_acce_var_ = 1e-4; // 加速度计零偏游走标准差

    // 其他配置
    bool update_bias_gyro_ = true; // 是否更新陀螺仪bias
    bool update_bias_acce_ = true; // 是否更新加速度计bias

private:
    double current_time_ = 0.0; // 当前时间

    /// 名义状态
    VecT p_ = VecT::Zero();
    VecT v_ = VecT::Zero();
    SO3 R_;
    VecT bg_ = VecT::Zero();
    VecT ba_ = VecT::Zero();
    VecT g_{0, 0, -9.8};

    /// 误差状态
    Vec18T dx_ = Vec18T::Zero();

    /// 协方差阵
    Mat18T cov_ = Mat18T::Identity();

    /// 噪声阵
    MotionNoiseT Q_ = MotionNoiseT::Zero();
};

using ESKFD = ESKF<double>;
using ESKFF = ESKF<float>;


template <typename S>
bool ESKF<S>::Predict(const IMU &imu) {
    assert(imu.timestamp_ >= current_time_);

    double dt = (imu.timestamp_ - current_time_) / 1e9;
    if (dt > (5 * imu_dt_) || dt < 0) {
        // 时间间隔不对，可能是第一个IMU数据，没有历史信息
        // std::cout << "\033[1;32mskip this imu because dt_ = \033[0m" << dt << std::endl;
        current_time_ = imu.timestamp_;
        return false;
    }

    // nominal state 递推
    VecT new_p = p_ + v_ * dt + 0.5 * (R_ * (imu.acce_ - ba_)) * dt * dt + 0.5 * g_ * dt * dt;
    VecT new_v = v_ + R_ * (imu.acce_ - ba_) * dt + g_ * dt;
    SO3 new_R = R_ * SO3::exp((imu.gyro_ - bg_) * dt);

    R_ = new_R;
    v_ = new_v;
    p_ = new_p;

    // 其余状态维度不变

    // error state 递推
    // 计算运动过程雅可比矩阵 F
    Mat18T F = Mat18T::Identity();                                                 // 主对角线
    F.template block<3, 3>(0, 3) = Mat3T::Identity() * dt;                         // p 对 v
    F.template block<3, 3>(3, 6) = -R_.matrix() * SO3::hat(imu.acce_ - ba_) * dt;  // v对theta
    F.template block<3, 3>(3, 12) = -R_.matrix() * dt;                             // v 对 ba
    F.template block<3, 3>(3, 15) = Mat3T::Identity() * dt;                        // v 对 g
    F.template block<3, 3>(6, 6) = SO3::exp(-(imu.gyro_ - bg_) * dt).matrix();     // theta 对 theta
    F.template block<3, 3>(6, 9) = -Mat3T::Identity() * dt;                        // theta 对 bg

    // mean and cov prediction
    dx_ = F * dx_;
    cov_ = F * cov_.eval() * F.transpose() + Q_;
    current_time_ = imu.timestamp_;

    return true;
}

template <typename S>
bool ESKF<S>::ObserveSE3(const SE3& pose, double ang_noise, double x_noise, double y_noise) {
    /// 既有旋转，也有平移
    /// 观测状态变量中的p, R，H为6x18，其余为零
    Eigen::Matrix<S, 6, 18> H = Eigen::Matrix<S, 6, 18>::Zero();
    H.template block<3, 3>(0, 0) = Mat3T::Identity();  // P部分
    H.template block<3, 3>(3, 6) = Mat3T::Identity();  // R部分（3.66)

    // 卡尔曼增益和更新过程
    Vec6d noise_vec;
    noise_vec << x_noise, y_noise, 1e-10, ang_noise, 1e-10, 1e-10;

    Mat6d V = noise_vec.asDiagonal();
    Eigen::Matrix<S, 18, 6> K = cov_ * H.transpose() * (H * cov_ * H.transpose() + V).inverse();

    // 更新x和cov
    Vec6d innov = Vec6d::Zero();
    innov.template head<3>() = (pose.translation() - p_);          // 平移部分
    innov.template tail<3>() = (R_.inverse() * pose.so3()).log();  // 旋转部分(3.67)

    dx_ = K * innov;
    cov_ = (Mat18T::Identity() - K * H) * cov_;

    UpdateAndReset();
    return true;
}

#endif //ESKF_H