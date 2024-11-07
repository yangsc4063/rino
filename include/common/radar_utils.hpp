//
// Created by Yangsc on 23-9-20.
//

#ifndef RADAR_UTILS_H
#define RADAR_UTILS_H

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <csignal>

#include <boost/algorithm/string.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>


// Radar 读数
struct Radar {
    cv::Mat raw_data_;
    cv::Mat img_in_cart_;

    int64_t              timestamp_;
    std::vector<int64_t> point_times_;

    // All the sizes of the below member variables have to be same!!!
    cv::Mat                   desc_;
    std::vector<cv::KeyPoint> kp_;
    Eigen::MatrixXd           feat_in_cart_;
    Eigen::MatrixXd           feat_in_polar_;
};

using RadarPtr = std::shared_ptr<Radar>;


inline void draw_points(const cv::Mat &cart_img, 
                        const std::vector<cv::KeyPoint> &bev_points,
                        cv::Mat &vis, 
                        std::vector<uint32_t> color = {(255), (0), (0)}) {
    cv::cvtColor(cart_img, vis, cv::COLOR_GRAY2BGR);
    const double radius = 3;
    for (cv::KeyPoint p: bev_points) {
        if (p.pt.x < radius || p.pt.x > cart_img.cols - radius || 
            p.pt.y > cart_img.rows - radius || p.pt.y < radius) { continue; }
        cv::circle(vis, p.pt, radius, cv::Scalar(color[0], color[1], color[2]), -1);
        if (cart_img.depth() == CV_8UC1)
            vis.at<cv::Vec3b>(int(p.pt.y), int(p.pt.x)) = cv::Vec3b(color[0], color[1], color[2]);
        if (cart_img.depth() == CV_32F)
            vis.at<cv::Vec3f>(int(p.pt.y), int(p.pt.x)) = cv::Vec3f(color[0], color[1], color[2]);
    }
}

inline void draw_points(const cv::Mat &cart_img, 
                        Eigen::MatrixXd cart_targets, 
                        float cart_resolution, int cart_pixel_width, 
                        cv::Mat &vis, 
                        std::vector<uint> color = {(255), (0), (0)}) {
    std::vector<cv::Point2f> bev_points;
    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;
    bev_points.clear();
    int       j = 0;
    for (uint i = 0; i < cart_targets.cols(); ++i) {
        double u = (cart_min_range + cart_targets(1, i)) / cart_resolution;
        double v = (cart_min_range - cart_targets(0, i)) / cart_resolution;
        if (0 < u && u < cart_pixel_width && 0 < v && v < cart_pixel_width) {
            bev_points.push_back(cv::Point2f(u, v));
            cart_targets(0, j) = cart_targets(0, i);
            cart_targets(1, j) = cart_targets(1, i);
            j++;
        }
    }
    cart_targets.conservativeResize(3, bev_points.size());

    cv::cvtColor(cart_img, vis, cv::COLOR_GRAY2BGR);
    const double radius = 2;
    for (cv::Point2f p: bev_points) {
        if (p.x < radius || p.x > cart_img.cols - radius || 
            p.y > cart_img.rows - radius || p.y < radius) { continue; }
        cv::circle(vis, p, radius, cv::Scalar(color[0], color[1], color[2]), -1);
        if (cart_img.depth() == CV_8UC1)
            vis.at<cv::Vec3b>(int(p.y), int(p.x)) = cv::Vec3b(color[0], color[1], color[2]);
        if (cart_img.depth() == CV_32F)
            vis.at<cv::Vec3f>(int(p.y), int(p.x)) = cv::Vec3f(color[0], color[1], color[2]);
    }
}

inline void draw_points(const cv::Mat &cart_img, 
                        Eigen::MatrixXd cart_targets_1,
                        Eigen::MatrixXd cart_targets_2,
                        float cart_resolution, int cart_pixel_width, 
                        cv::Mat &vis, 
                        std::vector<uint> color_1 = {(255), (0), (0)},
                        std::vector<uint> color_2 = {(0), (0), (255)}) {
    std::vector<cv::Point2f> bev_points_1;
    std::vector<cv::Point2f> bev_points_2;
    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;
    bev_points_1.clear();
    int       j = 0;
    for (uint i = 0; i < cart_targets_1.cols(); ++i) {
        double u_1 = (cart_min_range + cart_targets_1(1, i)) / cart_resolution;
        // double u_1 = (cart_min_range - cart_targets_1(1, i)) / cart_resolution;
        double v_1 = (cart_min_range - cart_targets_1(0, i)) / cart_resolution;
        if (0 < u_1 && u_1 < cart_pixel_width && 0 < v_1 && v_1 < cart_pixel_width) {
            bev_points_1.push_back(cv::Point2f(u_1, v_1));
            cart_targets_1(0, j) = cart_targets_1(0, i);
            cart_targets_1(1, j) = cart_targets_1(1, i);
            j++;
        }
    }
    cart_targets_1.conservativeResize(3, bev_points_1.size());
    bev_points_2.clear();
    j = 0;
    for (uint i = 0; i < cart_targets_2.cols(); ++i) {
        double u_2 = (cart_min_range - cart_targets_2(1, i)) / cart_resolution;
        double v_2 = (cart_min_range - cart_targets_2(0, i)) / cart_resolution;
        if (0 < u_2 && u_2 < cart_pixel_width && 0 < v_2 && v_2 < cart_pixel_width) {
            bev_points_2.push_back(cv::Point2f(u_2, v_2));
            cart_targets_2(0, j) = cart_targets_2(0, i);
            cart_targets_2(1, j) = cart_targets_2(1, i);
            j++;
        }
    }
    cart_targets_2.conservativeResize(3, bev_points_2.size());

    cv::cvtColor(cart_img, vis, cv::COLOR_GRAY2BGR);
    const double radius = 2;
    for (cv::Point2f p: bev_points_1) {
        if (p.x < radius || p.x > cart_img.cols - radius || 
            p.y > cart_img.rows - radius || p.y < radius) { continue; }
        cv::circle(vis, p, radius, cv::Scalar(color_1[0], color_1[1], color_1[2]), -1);
        if (cart_img.depth() == CV_8UC1)
            vis.at<cv::Vec3b>(int(p.y), int(p.x)) = cv::Vec3b(color_1[0], color_1[1], color_1[2]);
        if (cart_img.depth() == CV_32F)
            vis.at<cv::Vec3f>(int(p.y), int(p.x)) = cv::Vec3f(color_1[0], color_1[1], color_1[2]);
    }
    for (cv::Point2f p: bev_points_2) {
        if (p.x < radius || p.x > cart_img.cols - radius || 
            p.y > cart_img.rows - radius || p.y < radius) { continue; }
        cv::circle(vis, p, radius, cv::Scalar(color_2[0], color_2[1], color_2[2]), -1);
        if (cart_img.depth() == CV_8UC1)
            vis.at<cv::Vec3b>(int(p.y), int(p.x)) = cv::Vec3b(color_2[0], color_2[1], color_2[2]);
        if (cart_img.depth() == CV_32F)
            vis.at<cv::Vec3f>(int(p.y), int(p.x)) = cv::Vec3f(color_2[0], color_2[1], color_2[2]);
    }
}

inline void draw_arrows(const cv::Mat &cart_img, 
                        Eigen::MatrixXd cart_targets_1,
                        Eigen::MatrixXd cart_targets_2,
                        float cart_resolution, int cart_pixel_width, 
                        cv::Mat &vis, 
                        std::vector<uint> color_1 = {(255), (0), (0)},
                        std::vector<uint> color_2 = {(0), (0), (255)}) {
    std::vector<cv::Point2f> bev_points_1;
    std::vector<cv::Point2f> bev_points_2;
    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;
    bev_points_1.clear();
    int       j = 0;
    for (uint i = 0; i < cart_targets_1.cols(); ++i) {
        double u_1 = (cart_min_range + cart_targets_1(1, i)) / cart_resolution;
        double v_1 = (cart_min_range - cart_targets_1(0, i)) / cart_resolution;
        if (0 < u_1 && u_1 < cart_pixel_width && 0 < v_1 && v_1 < cart_pixel_width) {
            bev_points_1.push_back(cv::Point2f(u_1, v_1));
            cart_targets_1(0, j) = cart_targets_1(0, i);
            cart_targets_1(1, j) = cart_targets_1(1, i);
            j++;
        }
    }
    cart_targets_1.conservativeResize(3, bev_points_1.size());
    bev_points_2.clear();
    j = 0;
    for (uint i = 0; i < cart_targets_2.cols(); ++i) {
        double u_2 = (cart_min_range + cart_targets_2(1, i)) / cart_resolution;
        double v_2 = (cart_min_range - cart_targets_2(0, i)) / cart_resolution;
        if (0 < u_2 && u_2 < cart_pixel_width && 0 < v_2 && v_2 < cart_pixel_width) {
            bev_points_2.push_back(cv::Point2f(u_2, v_2));
            cart_targets_2(0, j) = cart_targets_2(0, i);
            cart_targets_2(1, j) = cart_targets_2(1, i);
            j++;
        }
    }
    cart_targets_2.conservativeResize(3, bev_points_2.size());

    cv::cvtColor(cart_img, vis, cv::COLOR_GRAY2BGR);
    const double radius = 2;
    for (cv::Point2f p: bev_points_1) {
        if (p.x < radius || p.x > cart_img.cols - radius || 
            p.y > cart_img.rows - radius || p.y < radius) { continue; }
        cv::circle(vis, p, radius, cv::Scalar(color_1[0], color_1[1], color_1[2]), -1);
        if (cart_img.depth() == CV_8UC1)
            vis.at<cv::Vec3b>(int(p.y), int(p.x)) = cv::Vec3b(color_1[0], color_1[1], color_1[2]);
        if (cart_img.depth() == CV_32F)
            vis.at<cv::Vec3f>(int(p.y), int(p.x)) = cv::Vec3f(color_1[0], color_1[1], color_1[2]);
        cv::Point2f ori_p(cart_min_range / cart_resolution, cart_min_range / cart_resolution);
        cv::Point2f end_p(p.x, p.y);
        cv::arrowedLine(vis, ori_p, end_p, cv::Scalar(color_1[0], color_1[1], color_1[2]), 1, cv::LINE_AA, 0, 0.05);
    }
    for (cv::Point2f p: bev_points_2) {
        if (p.x < radius || p.x > cart_img.cols - radius || 
            p.y > cart_img.rows - radius || p.y < radius) { continue; }
        cv::circle(vis, p, radius, cv::Scalar(color_2[0], color_2[1], color_2[2]), -1);
        if (cart_img.depth() == CV_8UC1)
            vis.at<cv::Vec3b>(int(p.y), int(p.x)) = cv::Vec3b(color_2[0], color_2[1], color_2[2]);
        if (cart_img.depth() == CV_32F)
            vis.at<cv::Vec3f>(int(p.y), int(p.x)) = cv::Vec3f(color_2[0], color_2[1], color_2[2]);
        cv::Point2f ori_p(cart_min_range / cart_resolution, cart_min_range / cart_resolution);
        cv::Point2f end_p(p.x, p.y);
        cv::arrowedLine(vis, ori_p, end_p, cv::Scalar(color_2[0], color_2[1], color_2[2]), 1, cv::LINE_AA, 0, 0.05);
    }
}

#endif  // RADAR_UTILS_H