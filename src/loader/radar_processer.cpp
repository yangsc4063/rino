//
// Created by Yangsc on 23-9-26.
//

#include <cmath>

#include "loader/radar_processer.hpp"


using MeasMatching = std::tuple<MeasureGroup, MeasureGroup>;

double get_azimuth_index(std::vector<double> &azimuths, double azimuth) {
    double    mind    = 1000;
    double    closest = 0;
    int       M       = azimuths.size();
    for (uint i       = 0; i < azimuths.size(); ++i) {
        double d = fabs(azimuths[i] - azimuth);
        if (d < mind) {
            mind    = d;
            closest = i;
        }
    }
    if (azimuths[closest] < azimuth) {
        double delta = 0;
        if (closest < M - 1)
            delta    = (azimuth - azimuths[closest]) / (azimuths[closest + 1] - azimuths[closest]);
        closest += delta;
    } else if (azimuths[closest] > azimuth) {
        double delta = 0;
        if (closest > 0)
            delta = (azimuths[closest] - azimuth) / (azimuths[closest] - azimuths[closest - 1]);
        closest -= delta;
    }
    return closest;
}

void polar_to_cartesian_radar_img(std::vector<double> &azimuths, 
                                  cv::Mat &raw_data, 
                                  float radar_resolution,
                                  float cart_resolution, 
                                  int cart_pixel_width, 
                                  bool interpolate_crossover,
                                  cv::Mat &cart_img, 
                                  int output_type) {

    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;

    cv::Mat map_x = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat map_y = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);

#pragma omp parallel for collapse(2)
    for (int j = 0; j < map_y.cols; ++j) {
        for (int i = 0; i < map_y.rows; ++i) {
            map_y.at<float>(i, j) = -1 * cart_min_range + j * cart_resolution;
        }
    }
#pragma omp parallel for collapse(2)
    for (int i     = 0; i < map_x.rows; ++i) {
        for (int j = 0; j < map_x.cols; ++j) {
            map_x.at<float>(i, j) = cart_min_range - i * cart_resolution;
        }
    }
    cv::Mat  range = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat  angle = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);

    double azimuth_step = azimuths[1] - azimuths[0];
#pragma omp parallel for collapse(2)
    for (int i = 0; i < range.rows; ++i) {
        for (int j = 0; j < range.cols; ++j) {
            float x = map_x.at<float>(i, j);
            float y = map_y.at<float>(i, j);
            float r = (sqrt(pow(x, 2) + pow(y, 2)) - radar_resolution / 2) / radar_resolution;
            if (r < 0)
                r = 0;
            range.at<float>(i, j) = r;
            float theta = atan2f(y, x);
            if (theta < 0)
                theta += 2 * M_PI;
            angle.at<float>(i, j) = get_azimuth_index(azimuths, theta);
        }
    }
    if (interpolate_crossover) {
        cv::Mat  a0   = cv::Mat::zeros(1, raw_data.cols, CV_32F);
        cv::Mat  aN_1 = cv::Mat::zeros(1, raw_data.cols, CV_32F);
        for (int j    = 0; j < raw_data.cols; ++j) {
            a0.at<float>(0, j)   = raw_data.at<float>(0, j);
            aN_1.at<float>(0, j) = raw_data.at<float>(raw_data.rows - 1, j);
        }
        cv::vconcat(aN_1, raw_data, raw_data);
        cv::vconcat(raw_data, a0, raw_data);
        angle = angle + 1;
    }
    cv::remap(raw_data, cart_img, range, angle, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    if (output_type == CV_8UC1) {
        double min, max;
        cv::minMaxLoc(cart_img, &min, &max);
        cart_img.convertTo(cart_img, CV_8UC1, 255.0 / max);
    }
}

void polar_to_cartesian_points(std::vector<double> azimuths,
                               float radar_resolution, 
                               Eigen::MatrixXd polar_points, Eigen::MatrixXd &cart_points) {
    cart_points = polar_points;
    for (size_t i = 0; i < polar_points.cols(); ++i) {
        double azimuth = azimuths[polar_points(0, i)];
        double r       = polar_points(1, i) * radar_resolution + radar_resolution / 2;
        cart_points(0, i) = r * cos(azimuth); // meter
        cart_points(1, i) = r * sin(azimuth); // meter
    }
}

void assign_timestamps(int64_t radar_start_time, int64_t radar_end_time, 
                       Eigen::MatrixXd &cart_points, 
                       std::vector<int64_t> &point_times) {
    point_times.resize(cart_points.cols());
    for (size_t i = 0; i < cart_points.cols(); ++i) {
        double angle = atan2(cart_points(1, i), cart_points(0, i));
        if (angle < 0)
            angle += 2 * M_PI;
        double time = radar_start_time + (radar_end_time - radar_start_time) * angle / (2 * M_PI);
        point_times[i] = time;
    }
}

// 体素化，输入：体素大小，输出：体素化后的点云
void voxelize(const double voxelization_size, Eigen::MatrixXd &cart_targets) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cart_tmp(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cart_tmp_voxel(new pcl::PointCloud<pcl::PointXYZ>);
    // 将Eigen矩阵转换为pcl点云
    eigen2pcl(cart_targets, *cart_tmp);
    /**
     * pcl::VoxelGrid 将原始点云中落入同一体素的点合并成一个代表该体素的点，
     * => 点云的降采样
     */ 
    static pcl::VoxelGrid<pcl::PointXYZ> voxel_filter; // 静态变量，只初始化一次
    voxel_filter.setInputCloud(cart_tmp);
    voxel_filter.setLeafSize(voxelization_size, voxelization_size, voxelization_size);
    voxel_filter.filter(*cart_tmp_voxel);
    int num_pts = cart_tmp_voxel->points.size();
    // 重置cart_targets矩阵
    cart_targets = Eigen::MatrixXd::Zero(3, num_pts);
    for (int ii = 0; ii < num_pts; ++ii) {
        const auto &pt = (*cart_tmp_voxel).points[ii];
        cart_targets(0, ii) = pt.x;
        cart_targets(1, ii) = pt.y;
    }
}

void convert_to_bev(Eigen::MatrixXd &cart_points, 
                    float cart_resolution, 
                    int cart_pixel_width, 
                    int patch_size,
                    std::vector<cv::KeyPoint> &bev_points, 
                    std::vector<int64_t> &point_times) {
    // 计算bev_points中的范围
    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;
    bev_points.clear();
    // 初始化索引 j 用于记录bev_points点的数量
    int j = 0;
    for (uint i = 0; i < cart_points.cols(); ++i) {
        // 计算点在bev_points中的像素坐标 u 和 v
        double u = (cart_min_range + cart_points(1, i)) / cart_resolution;
        double v = (cart_min_range - cart_points(0, i)) / cart_resolution;
        if (0 < u - patch_size && u + patch_size < cart_pixel_width && 
            0 < v - patch_size && v + patch_size < cart_pixel_width) {
            bev_points.push_back(cv::KeyPoint(u, v, patch_size));
            point_times[j] = point_times[i];
            cart_points(0, j) = cart_points(0, i);
            cart_points(1, j) = cart_points(1, i);
            j++;
        }
    }
    point_times.resize(bev_points.size());
    cart_points.conservativeResize(3, bev_points.size());
}

MeasureGroup RadarProcesser::extractFeatures(const MeasureGroup &meas) {

    // cv::imshow("Test for raw_data", raw_data);
    // cv::waitKey(20);

    std::vector<int64_t> times;
    std::vector<double>  azimuths;
    cv::Mat raw_data = meas.radar_->raw_data_;

    // 为每一角度分配时间戳
    int64_t radar_start_time = meas.radar_start_time_;
    int64_t radar_end_time   = meas.radar_end_time_;

    // 求每一角度
    double intv_az = 2 * M_PI / azimuth_num_;

    for (int i = 0; i < azimuth_num_; ++i) {
        double a = (i + 1/2) * intv_az;
        azimuths.push_back(a);
    }

    MeasureGroup curr_meas = meas;

    if (keypoint_extraction_ == "cen2018") {
        cen2018feat(raw_data, zq_, sigma_gauss_, min_range_, curr_meas.radar_->feat_in_polar_);
    }
    else if (keypoint_extraction_ == "cen2019") {
        cen2019feat(raw_data, max_points_, min_range_, curr_meas.radar_->feat_in_polar_);
    }
    else if (keypoint_extraction_ == "yang2024") {
        yang2024feat(raw_data, zq_, sigma_gauss_, min_range_, max_range_, curr_meas.radar_->feat_in_polar_);
    }

    polar_to_cartesian_radar_img(azimuths, raw_data, 
                                 radar_resolution_, cart_resolution_, cart_pixel_width_, use_interpolation_,
                                 curr_meas.radar_->img_in_cart_, 
                                 CV_8UC1);

    // cv::imshow("Test for img_in_cart_", curr_meas.radar_->img_in_cart_);
    // cv::waitKey(20);

    polar_to_cartesian_points(azimuths,
                              radar_resolution_,
                              curr_meas.radar_->feat_in_polar_, curr_meas.radar_->feat_in_cart_);

    /**
     * Q. Why voxelization?
     * => It dramatically improves the performance!
     */
    // 体素化
    if (use_voxelization_) {
        voxelize(voxel_size_, curr_meas.radar_->feat_in_cart_);
    }

    assign_timestamps(radar_start_time, radar_end_time, 
                      curr_meas.radar_->feat_in_cart_, 
                      curr_meas.radar_->point_times_);
    
    convert_to_bev(curr_meas.radar_->feat_in_cart_, 
                   cart_resolution_, cart_pixel_width_, patch_size_, 
                   curr_meas.radar_->kp_, // 以cv::KeyPoint类型存储关键点
                   curr_meas.radar_->point_times_);
    // cv::Mat back_ground;
    // draw_points(curr_meas.radar_->img_in_cart_, curr_meas.radar_->feat_in_cart_, 0.2592, 964, back_ground, {255, 0, 0});
    // cv::imshow("Test for feature extraction", back_ground);
    // cv::waitKey(20);

    detector_->compute(curr_meas.radar_->img_in_cart_, curr_meas.radar_->kp_, curr_meas.radar_->desc_); // 计算描述子

    return curr_meas;
};

MeasMatching RadarProcesser::matchFeatures(const MeasureGroup &prev_meas, const MeasureGroup &curr_meas) {
    MeasureGroup prev_matched, curr_matched;
    prev_matched.radar_start_time_ = prev_meas.radar_start_time_;
    curr_matched.radar_end_time_ = curr_meas.radar_end_time_;
    prev_matched.radar_ = std::make_shared<Radar>();
    curr_matched.radar_ = std::make_shared<Radar>();
    *prev_matched.radar_ = *prev_meas.radar_;
    *curr_matched.radar_ = *curr_meas.radar_;
    prev_matched.imus_ = prev_meas.imus_;
    curr_matched.imus_ = curr_meas.imus_;

    std::vector<cv::DMatch> good_matches;
    good_matches.reserve(curr_meas.radar_->feat_in_cart_.cols());

    std::vector<std::vector<cv::DMatch>> knn_matches;
    knn_matches.reserve(curr_meas.radar_->feat_in_cart_.cols());
    // Match keypoint descriptors
    matcher_->knnMatch(prev_meas.radar_->desc_, curr_meas.radar_->desc_, knn_matches, 2);

    // Filter matches using nearest neighbor distance ratio (Lowe, Szeliski)
    for (uint j = 0; j < knn_matches.size(); ++j) {
        if (!knn_matches[j].size())
            continue;
        float nndr_ = 0.8;
        if (knn_matches[j][0].distance < nndr_ * knn_matches[j][1].distance) {
            auto &idx_pair = knn_matches[j][0];
            good_matches.emplace_back(idx_pair);
        }
    }

    int num_matches = good_matches.size();
    // Convert the good key point matches to Eigen matrices
    prev_matched.radar_->feat_in_cart_.resize(3, num_matches);
    curr_matched.radar_->feat_in_cart_.resize(3, num_matches);
    prev_matched.radar_->point_times_.resize(num_matches);
    curr_matched.radar_->point_times_.resize(num_matches);
    prev_matched.radar_->kp_.resize(num_matches);
    curr_matched.radar_->kp_.resize(num_matches);

    for (uint j = 0; j < num_matches; ++j) {
        prev_matched.radar_->feat_in_cart_.col(j) = prev_meas.radar_->feat_in_cart_.block<3, 1>(0, good_matches[j].queryIdx);
        curr_matched.radar_->feat_in_cart_.col(j) = curr_meas.radar_->feat_in_cart_.block<3, 1>(0, good_matches[j].trainIdx);

        prev_matched.radar_->point_times_[j] = prev_meas.radar_->point_times_[good_matches[j].queryIdx];
        curr_matched.radar_->point_times_[j] = curr_meas.radar_->point_times_[good_matches[j].trainIdx];

        // For visualization
        prev_matched.radar_->kp_[j] = prev_meas.radar_->kp_[good_matches[j].queryIdx];
        curr_matched.radar_->kp_[j] = curr_meas.radar_->kp_[good_matches[j].trainIdx];
    }
    // cv::Mat back_ground;
    // draw_points(curr_matched.radar_->img_in_cart_, curr_matched.radar_->feat_in_cart_, 0.2592, 964, back_ground, {255, 0, 0});
    // cv::imshow("Test for feature matching", back_ground);
    // cv::waitKey(20);
    return {prev_matched, curr_matched};
};