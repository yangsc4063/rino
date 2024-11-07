//
// Created by Yangsc on 23-10-10.
//

#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <numeric>

#include <common/eigen_types.hpp>
#include <common/nav_state.hpp>

// 常量定义
constexpr double kDEG2RAD = M_PI / 180.0;  // deg->rad
constexpr double kRAD2DEG = 180.0 / M_PI;  // rad -> deg
constexpr double G_m_s2 = 9.81;            // 重力大小

/**
 * 计算一个容器内数据的均值与对角形式协方差
 * @tparam C    容器类型
 * @tparam D    结果类型
 * @tparam Getter   获取数据函数, 接收一个容器内数据类型，返回一个D类型
 */
template <typename C, typename D, typename Getter>
void ComputeMeanAndCovDiag(const C& data, D& mean, D& cov_diag, Getter&& getter) {
    size_t len = data.size();
    assert(len > 1);
    // clang-format off
    mean = std::accumulate(data.begin(), data.end(), D::Zero().eval(),
                           [&getter](const D& sum, const auto& data) -> D { return sum + getter(data); }) / len;
    cov_diag = std::accumulate(data.begin(), data.end(), D::Zero().eval(),
                               [&mean, &getter](const D& sum, const auto& data) -> D {
                                   return sum + (getter(data) - mean).cwiseAbs2().eval();
                               }) / (len - 1);
    // clang-format on
}

/**
 * 计算一个容器内数据的均值与矩阵形式协方差
 * @tparam C    容器类型
 * @tparam int 　数据维度
 * @tparam Getter   获取数据函数, 接收一个容器内数据类型，返回一个Eigen::Matrix<double, dim,1> 矢量类型
 */
template <typename C, int dim, typename Getter>
void ComputeMeanAndCov(const C& data, Eigen::Matrix<double, dim, 1>& mean, Eigen::Matrix<double, dim, dim>& cov,
                       Getter&& getter) {
    using D = Eigen::Matrix<double, dim, 1>;
    using E = Eigen::Matrix<double, dim, dim>;
    size_t len = data.size();
    assert(len > 1);

    // clang-format off
    mean = std::accumulate(data.begin(), data.end(), Eigen::Matrix<double, dim, 1>::Zero().eval(),
                           [&getter](const D& sum, const auto& data) -> D { return sum + getter(data); }) / len;
    cov = std::accumulate(data.begin(), data.end(), E::Zero().eval(),
                          [&mean, &getter](const E& sum, const auto& data) -> E {
                              D v = getter(data) - mean;
                              return sum + v * v.transpose();
                          }) / (len - 1);
    // clang-format on
}

/**
 * pose 插值算法
 * @tparam T 数据类型
 * @tparam C 数据容器类型
 * @tparam FT 获取时间函数
 * @tparam FP 获取pose函数
 * @param query_time 查找时间
 * @param data  数据容器
 * @param take_pose_func 从数据中取pose的谓词，接受一个数据，返回一个SE3
 * @param result 查询结果
 * @param best_match_iter 查找到的最近匹配
 *
 * NOTE 要求query_time必须在data最大时间和最小时间之间(容许0.5s内误差)
 * data的map按时间排序
 * @return
 */
template <typename T, typename C, typename FT, typename FP>
inline bool PoseInterp(int64_t query_time, C&& data, 
                       FT&& take_time_func, FP&& take_pose_func, 
                       SE3& result, int64_t time_th = 0.5*1e9) {
    T best_match;

    if (data.empty()) {
        std::cout << "\033[1;31m[Warning]Cannot interp because data is empty.\033[0m" << std::endl;
        return false;
    }

    int64_t last_time = take_time_func(*data.rbegin());
    if (query_time > last_time) {
        if (query_time < (last_time + time_th)) {
            // 尚可接受
            result = take_pose_func(*data.rbegin());
            best_match = *data.rbegin();
            return true;
        }
        return false;
    }

    auto match_iter = data.begin();
    for (auto iter = data.begin(); iter != data.end(); ++iter) {
        auto next_iter = iter;
        next_iter++;

        if (take_time_func(*iter) < query_time && take_time_func(*next_iter) >= query_time) {
            match_iter = iter;
            break;
        }
    }

    auto match_iter_n = match_iter;
    match_iter_n++;

    int64_t dt = take_time_func(*match_iter_n) - take_time_func(*match_iter);
    int64_t s = (query_time - take_time_func(*match_iter)) / dt;  // s=0 时为第一帧，s=1时为next
    // 出现了dt为0的bug
    if (fabs(dt) < 1e-6) {
        best_match = *match_iter;
        result = take_pose_func(*match_iter);
        return true;
    }

    SE3 pose_first = take_pose_func(*match_iter);
    SE3 pose_next = take_pose_func(*match_iter_n);
    result = {pose_first.unit_quaternion().slerp(s, pose_next.unit_quaternion()),
              pose_first.translation() * (1 - s) + pose_next.translation() * s};
    best_match = s < 0.5 ? *match_iter : *match_iter_n;
    return true;
}

/**
 * velocity 插值算法
 * @tparam T 数据类型
 * @tparam C 数据容器类型
 * @tparam FT 获取时间函数
 * @tparam FV 获取velocity函数
 * @param query_time 查找时间
 * @param data 数据容器
 * @param take_time_func 从数据中取时间的谓词，接受一个数据，返回一个double
 * @param take_velocity_func 从数据中取velocity的谓词，接受一个数据，返回一个Vec3T
 * @param result 查询结果
 * 
 * NOTE 要求query_time必须在data最大时间和最小时间之间(容许0.5s内误差)
 * data的map按时间排序
 * @return
 */
template <typename T, typename C, typename FT, typename FV>
inline bool VelInterp(int64_t query_time, C &&data, 
                      FT &&take_time_func, FV &&take_vel_func, 
                      Vec3d &result, float time_th = 0.5*1e9) {
    T best_match;

    if (data.empty()) {
        std::cout << "\033[1;31m[Warning]Cannot interp because data is empty.\033[0m" << std::endl;
        return false;
    }

    int64_t last_time = take_time_func(*data.rbegin());
    if (query_time > last_time) {
        if (query_time < (last_time + time_th)) {
            // 尚可接受
            result = take_vel_func(*data.rbegin());
            best_match = *data.rbegin();
            return true;
        }
        return false;
    }

    auto match_iter = data.begin();
    for (auto iter = data.begin(); iter != data.end(); ++iter) {
        auto next_iter = iter;
        next_iter++;

        if (take_time_func(*iter) < query_time && take_time_func(*next_iter) >= query_time) {
            match_iter = iter;
            break;
        }
    }

    auto match_iter_n = match_iter;
    match_iter_n++;

    int64_t dt = take_time_func(*match_iter_n) - take_time_func(*match_iter);
    int64_t s = (query_time - take_time_func(*match_iter)) / dt;  // s=0 时为第一帧，s=1时为next
    // 出现了dt为0的bug
    if (fabs(dt) < 1e-6) {
        best_match = *match_iter;
        result = take_vel_func(*match_iter);
        return true;
    }

    Vec3d vel_first = take_vel_func(*match_iter);
    Vec3d vel_next = take_vel_func(*match_iter_n);
    result = vel_first * (1 - s) + vel_next * s;
    best_match = s < 0.5 ? *match_iter : *match_iter_n;
    return true;
}

#endif //MATH_UTILS_H