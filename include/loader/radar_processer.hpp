//
// Created by Yangsc on 23-9-25.
//

#ifndef RADAR_PROCESSING_H
#define RADAR_PROCESSING_H

#include <opencv2/features2d.hpp>

#include "common/measure_utils.hpp"
#include "common/conversion.hpp"
#include "feature_exac.hpp"


class RadarProcesser {
public:
    using MeasMatching = std::tuple<MeasureGroup, MeasureGroup>;

    RadarProcesser() {};

    // 设置Radar processer参数
    void SetParams(std::string dataset,
                   std::string keypoint_extraction,
                   double voxel_size,
                   bool use_voxelization) {
        dataset_ = dataset;
        keypoint_extraction_ = keypoint_extraction;
        voxel_size_ = voxel_size;
        use_voxelization_ = use_voxelization;

        // 设置dataset参数
        if (dataset_ == "mulran") {
            std::cout << "\033[1;32mParams of MulRan Dataset are set\033[0m" << std::endl;
            min_range_             = 58;           // 雷达的最小探测范围
            radar_resolution_      = 0.05952;      // 每个距离单元的雷达分辨率，以每个单元的米数表示
            cart_resolution_       = 0.2592;       // 每像素的米数
            cart_pixel_width_      = 964;          // 笛卡尔坐标图像的像素高度和宽度，以像素表示
        }
        // 设置dataset参数
        else if (dataset_ == "boreas") {
            std::cout << "\033[1;32mParams of Boreas Dataset are set\033[0m" << std::endl;
            min_range_             = 58;           // 雷达的最小探测范围
            radar_resolution_      = 0.05952;      // 每个距离单元的雷达分辨率，以每个单元的米数表示
            cart_resolution_       = 0.2592;       // 每像素的米数
            cart_pixel_width_      = 964;          // 笛卡尔坐标图像的像素高度和宽度，以像素表示
        }
        else { 
            throw std::invalid_argument("Note implemented!"); 
        }

        // 设置keypoint提取方法
        if (keypoint_extraction_ != "cen2018" && keypoint_extraction_ != "cen2019" && keypoint_extraction_ != "yang2024") {
            throw std::invalid_argument("Keypoint extraction method seems to be wrong!");
        }

        // 创建ORB特征提取器（结合了FAST关键点检测器、BRIEF特征描述子以及旋转不变性）
        detector_ = cv::ORB::create();
        detector_->setPatchSize(patch_size_);     // 设置ORB特征提取器的patch_size参数：控制关键点周围用来计算特征的区域的大小
        detector_->setEdgeThreshold(patch_size_); // 设置ORB特征提取器的edge_threshold参数：控制检测到的特征点是否靠近图像边缘
    }

    MeasureGroup extractFeatures(const MeasureGroup &meas);

    MeasMatching matchFeatures(const MeasureGroup &prev_meas, const MeasureGroup &curr_meas);

private:
    std::string dataset_;
    std::string keypoint_extraction_;

    int    azimuth_num_ = 400;
    int    min_range_;
    int    cart_pixel_width_;
    float  radar_resolution_;
    float  cart_resolution_;

    // cen2018 parameters
    float zq_          = 3.0;
    int   sigma_gauss_ = 17;
    // cen2019 parameters
    int   max_points_  = 1200; // WARNING! It directly affects the performance
    // yang2024 parameters
    int   max_range_   = 1500;

    // ORB descriptor / matching parameters
    cv::Ptr<cv::ORB> detector_;
    int              patch_size_ = 21;   // width of patch in pixels in cartesian radar image
    float            nndr_       = 0.80; // Nearest neighbor distance ratio

    double voxel_size_;

    bool is_initial_        = true;
    bool use_voxelization_  = true;
    bool use_interpolation_ = true;

    /**
     * BRUTEFORCE_HAMMING for ORB descriptors FLANNBASED for cen2019 descriptors
     * Hyungtae: However, empirically, ORB descriptor shows more precise results
     */
    cv::Ptr<cv::DescriptorMatcher> matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
};

#endif // RADAR_PROCESSING_H