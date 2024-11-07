//
// Created by Yangsc on 23-9-26.
//

#ifndef FEATURE_EXAC_HPP
#define FEATURE_EXAC_HPP

#include <iostream>
#include <chrono>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>


struct Point {
    float i;
    int   a;
    int   r;

    Point(float i_, int a_, int r_) {
        i = i_;
        a = a_;
        r = r_;
    }
};

/*!
   \brief Extract features from polar radar data using the method described in cen_icra18
   \param fft_data Polar radar power readings
   \param zq If y(i, j) > zq * sigma_q then it is considered a potential target point
   \param sigma_gauss std dev of the gaussian filter uesd to smooth the radar signal
   \param min_range We ignore the range bins less than this
   \param feat_in_polar [out] Matrix of feature locations (azimuth_bin, range_bin, 1) x N
*/
double cen2018feat(cv::Mat raw_data, float zq, int sigma_gauss, int min_range, Eigen::MatrixXd &feat_in_polar);

/*!
   \brief Extract features from polar radar data using the method described in cen_icra19
   \param fft_data Polar radar power readings
   \param max_points Maximum number of targets points to be extracted from the radar image
   \param min_range We ignore the range bins less than this
   \param feat_in_polar [out] Matrix of feature locations (azimuth_bin, range_bin, 1) x N
*/
double cen2019feat(cv::Mat raw_data, int max_points, int min_range, Eigen::MatrixXd &feat_in_polar);

double yang2024feat(cv::Mat raw_data, float zq, int sigma_gauss, int min_range, int max_range,
                    Eigen::MatrixXd &feat_in_polar);

#endif //FEATURE_EXAC_HPP