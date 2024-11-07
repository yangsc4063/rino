//
// Created by Yangsc on 23-9-27.
//

#include "loader/feature_exac.hpp"


double cen2018feat(cv::Mat raw_data, float zq, int sigma_gauss, int min_range, 
                   Eigen::MatrixXd &feat_in_polar) {
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<float> sigma_q(raw_data.rows, 0);
    // Estimate the bias (计算每一行的均值) and subtract it from the signal
    cv::Mat            q = raw_data.clone();
    for (int           i = 0; i < raw_data.rows; ++i) {
        float    mean = 0;
        for (int j    = 0; j < raw_data.cols; ++j) {
            mean += raw_data.at<float>(i, j);
        }
        mean /= raw_data.cols;
        for (int j = 0; j < raw_data.cols; ++j) {
            q.at<float>(i, j) = raw_data.at<float>(i, j) - mean;
        }
    }

    // Create 1D Gaussian Filter (0.09)
    assert(sigma_gauss % 2 == 1);
    int      fsize   = sigma_gauss * 3;
    int      mu      = fsize / 2;
    float    sig_sqr = sigma_gauss * sigma_gauss;
    cv::Mat  filter  = cv::Mat::zeros(1, fsize, CV_32F);
    float    s       = 0;
    for (int i       = 0; i < fsize; ++i) {
        filter.at<float>(0, i) = exp(-0.5 * (i - mu) * (i - mu) / sig_sqr);
        s += filter.at<float>(0, i);
    }
    filter /= s;
    cv::Mat p;
    // 将滤波器应用于q，并将结果存储于p
    cv::filter2D(q, p, -1, filter, cv::Point(-1, -1), 0, cv::BORDER_REFLECT101);

    // Estimate variance of noise at each azimuth (0.004)
    for (int i = 0; i < raw_data.rows; ++i) {
        int      nonzero = 0;
        for (int j       = 0; j < raw_data.cols; ++j) {
            float n = q.at<float>(i, j);
            if (n < 0) {
                sigma_q[i] += 2 * (n * n);
                nonzero++;
            }
        }
        if (nonzero)
            sigma_q[i]   = sqrt(sigma_q[i] / nonzero);
        else
            sigma_q[i] = 0.034;
    }

    // Extract peak centers from each azimuth
    std::vector<std::vector<cv::Point2f>> t(raw_data.rows);
#pragma omp parallel for
    for (int i = 0; i < raw_data.rows; ++i) {
        std::vector<int> peak_points;
        float            thres = zq * sigma_q[i];
        for (int         j     = min_range; j < raw_data.cols; ++j) {
            float nqp = exp(-0.5 * pow((q.at<float>(i, j) - p.at<float>(i, j)) / sigma_q[i], 2));
            float npp = exp(-0.5 * pow(p.at<float>(i, j) / sigma_q[i], 2));
            float b   = nqp - npp;
            float y   = q.at<float>(i, j) * (1 - nqp) + p.at<float>(i, j) * b;
            if (y > thres) {
                peak_points.push_back(j);
            } else if (peak_points.size() > 0) {
                t[i].push_back(cv::Point(i, peak_points[peak_points.size() / 2]));
                peak_points.clear();
            }
        }
        if (peak_points.size() > 0)
            t[i].push_back(cv::Point(i, peak_points[peak_points.size() / 2]));
    }

    int       size = 0;
    for (uint i    = 0; i < t.size(); ++i) {
        size += t[i].size();
    }
    feat_in_polar        = Eigen::MatrixXd::Ones(3, size);
    int       k = 0;
    for (uint i = 0; i < t.size(); ++i) {
        for (uint j = 0; j < t[i].size(); ++j) {
            feat_in_polar(0, k) = t[i][j].x;
            feat_in_polar(1, k) = t[i][j].y;
            k++;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> e  = t2 - t1;
    return e.count();
}


struct greater_than_pt {
    inline bool operator()(const Point &p1, const Point &p2) {
        return p1.i > p2.i;
    }
};

static void findRangeBoundaries(cv::Mat &s, int a, int r, int &r_low, int &r_high) {
    r_low  = r;
    r_high = r;
    if (r > 0) {
        for (int i = r - 1; i >= 0; i--) {
            if (s.at<float>(a, i) < 0) {
                r_low = i;
                break;
            }
            else
                continue;
        }
    }
    if (r < s.rows - 1) {
        for (int i = r + 1; i < s.cols; i++) {
            if (s.at<float>(a, i) < 0) {
                r_high = i;
                break;
            }
            else
                continue;
        }
    }
}

static bool checkAdjacentMarked(cv::Mat &R, int a, int start, int end) {
    int below = a - 1;
    int above = a + 1;
    if (below < 0)
        below = R.rows - 1;
    if (above >= R.rows)
        above = 0;
    for (int r = start; r <= end; r++) {
        if (R.at<float>(below, r) || R.at<float>(above, r))
            return true;
    }
    return false;
}

static void getMaxInRegion(cv::Mat &h, int a, int start, int end, int &max_r) {
    int max = -1000;
    for (int r = start; r <= end; r++) {
        if (h.at<float>(a, r) > max) {
            max   = h.at<float>(a, r);
            max_r = r;
        }
    }
}

double cen2019feat(cv::Mat raw_data, int max_points, int min_range, 
                   Eigen::MatrixXd &feat_in_polar) {
    auto t1 = std::chrono::high_resolution_clock::now();

    // prewitt滤波器的卷积核
    cv::Mat prewitt = cv::Mat::zeros(1, 3, CV_32F);
    prewitt.at<float>(0, 0) = -1;
    prewitt.at<float>(0, 2) = 1;

    cv::Mat g;
    cv::filter2D(raw_data, g, -1, prewitt, cv::Point(-1, -1), 0, cv::BORDER_REFLECT101);

    // 对图像进行归一化，确保所有像素值都在0到1之间
    g = cv::abs(g);
    double g_max = 1, g_min = 1;
    cv::minMaxIdx(g, &g_min, &g_max);
    g /= g_max;

    float   mean   = cv::mean(raw_data)[0];
    cv::Mat s      = raw_data - mean;
    cv::Mat h      = s.mul(1 - g); // 由Prewitt滤波器的响应对数据进行加权缩放
    float   mean_h = cv::mean(h)[0];

    // Get indices in descending order of intensity
    std::vector<Point> vec;
    for (int i = 0; i < raw_data.rows; ++i) {
        for (int j = 0; j < raw_data.cols; ++j) {
            // 判断 h 矩阵中的值是否大于之前计算的平均值 mean_h。
            if (h.at<float>(i, j) > mean_h)
                // 将当前点的信息（h(i,j)、行号和列号）构造成一个 Point 对象，并将该对象添加到 vec 向量中。
                vec.push_back(Point(h.at<float>(i, j), i, j));
        }
    }
    // 排序自定义的比较函数 greater_than_pt() 表示按 h(i,j) 大小排序
    std::sort(vec.begin(), vec.end(), greater_than_pt());

    // Create a matrix, R, of "marked" regions consisting of continuous regions of an azimuth that may contain a target
    int     false_count = raw_data.rows * raw_data.cols;
    uint    j           = 0;
    int     l           = 0;
    cv::Mat R           = cv::Mat::zeros(raw_data.rows, raw_data.cols, CV_32F);
    while (l < max_points && j < vec.size() && false_count > 0) {
        // 检查矩阵 R 中的特定位置 (vec[j].a, vec[j].r) 是否已经被标记，若没有则开始循环。
        if (!R.at<float>(vec[j].a, vec[j].r)) {
            int r_low  = vec[j].r;
            int r_high = vec[j].r;
            findRangeBoundaries(s, vec[j].a, vec[j].r, r_low, r_high);
            bool already_marked = false;
            for (int i = r_low; i <= r_high; i++) {
                if (R.at<float>(vec[j].a, i)) {
                    already_marked = true;
                    continue;
                }
                R.at<float>(vec[j].a, i) = 1;
                false_count--;
            }
            if (!already_marked)
                l++;
        }
        j++;
    } // 找到可能包含目标的 region，并将这些 region 标记为1，并使用矩阵 R 表示 marked region mask。

    std::vector<std::vector<cv::Point2f>> t(raw_data.rows);

#pragma omp parallel for
    for (int i = 0; i < raw_data.rows; i++) {
        // Find the continuous marked regions in each azimuth
        int  start    = 0;
        int  end      = 0;
        bool counting = false;
        for (int j = min_range; j < raw_data.cols; j++) {
            if (R.at<float>(i, j)) {
                if (!counting) {
                    start    = j;
                    end      = j;
                    counting = true;
                } 
                else {
                    end = j;
                }
            }
            else if (counting) {
                // Check whether adjacent azimuths contain a marked pixel in this range region
                if (checkAdjacentMarked(R, i, start, end)) { // 判断相邻方位角中是否包含在这个范围区域内有标记像素。
                    int max_r = start;
                    // 在给定范围内找到 h 矩阵中的最大值，并将最大值的行索引存储在 max_r 中
                    getMaxInRegion(h, i, start, end, max_r);
                    t[i].push_back(cv::Point(i, max_r));
                }
                counting = false;
            }
        }
    }

    int size = 0;
    for (uint i = 0; i < t.size(); ++i) {
        size += t[i].size();
    }
    feat_in_polar = Eigen::MatrixXd::Zero(3, size);
    int k = 0;
    for (uint i = 0; i < t.size(); ++i) {
        for (uint j = 0; j < t[i].size(); ++j) {
            feat_in_polar(0, k) = t[i][j].x;
            feat_in_polar(1, k) = t[i][j].y;
            k++;
        }
    }

    // std::cout << "feature size is " << size << std::endl;

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> e = t2 - t1;
    return e.count();
}

double yang2024feat(cv::Mat raw_data, float zq, int sigma_gauss, int min_range, int max_range,
                    Eigen::MatrixXd &feat_in_polar) {
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<float> sigma_q(raw_data.rows, 0);
    // Estimate the bias (计算每一行的均值) and subtract it from the signal
    cv::Mat            q = raw_data.clone();
    for (int           i = 0; i < raw_data.rows; ++i) {
        float    mean = 0;
        for (int j    = 0; j < raw_data.cols; ++j) {
            mean += raw_data.at<float>(i, j);
        }
        mean /= raw_data.cols;
        for (int j = 0; j < raw_data.cols; ++j) {
            q.at<float>(i, j) = raw_data.at<float>(i, j) - mean;
        }
    }

    // Create 1D Gaussian Filter (0.09)
    assert(sigma_gauss % 2 == 1);
    int      fsize   = sigma_gauss * 3;
    int      mu      = fsize / 2;
    float    sig_sqr = sigma_gauss * sigma_gauss;
    cv::Mat  filter  = cv::Mat::zeros(1, fsize, CV_32F);
    float    s       = 0;
    for (int i       = 0; i < fsize; ++i) {
        filter.at<float>(0, i) = exp(-0.5 * (i - mu) * (i - mu) / sig_sqr);
        s += filter.at<float>(0, i);
    }
    filter /= s;
    cv::Mat p;
    // 将滤波器应用于q，并将结果存储于p
    cv::filter2D(q, p, -1, filter, cv::Point(-1, -1), 0, cv::BORDER_REFLECT101);

    for (int i = 0; i < fsize; ++i) {
        filter.at<float>(0, i) = - 1 / (fsize + 1);
    }
    filter.at<float>(0, mu) = 1;
    cv::Mat o;
    // 将滤波器应用于q，并将结果存储于o
    cv::filter2D(q, o, -1, filter, cv::Point(-1, -1), 0, cv::BORDER_REFLECT101);

    // Estimate variance of noise at each azimuth (0.004)
    for (int i = 0; i < raw_data.rows; ++i) {
        int      nonzero = 0;
        for (int j       = 0; j < raw_data.cols; ++j) {
            float n = q.at<float>(i, j);
            if (n < 0) {
                sigma_q[i] += 2 * (n * n);
                nonzero++;
            }
        }
        if (nonzero)
            sigma_q[i] = sqrt(sigma_q[i] / nonzero);
        else
            sigma_q[i] = 0.034;
    }

    // Extract peak centers from each azimuth
    std::vector<std::vector<cv::Point2f>> t(raw_data.rows);
    // std::vector<Point> vec;
// #pragma omp parallel for
    for (int i = 0; i < raw_data.rows; ++i) {
        std::vector<int> peak_points;
        float            thres = zq * sigma_q[i];
        for (int         j     = min_range; j < max_range; ++j) {
            float nop = exp(-0.5 * pow((q.at<float>(i, j) - p.at<float>(i, j)) / sigma_q[i], 2));
            float npp = exp(-0.5 * pow(p.at<float>(i, j) / sigma_q[i], 2));
            float b   = nop - npp;
            float y   = o.at<float>(i, j) * (1 - nop) + p.at<float>(i, j) * b;
            if (y > thres) {
                peak_points.push_back(j);
            } 
            else if (peak_points.size() > 0) {
                // vec.push_back(Point(y, i, peak_points[peak_points.size() / 2]));
                t[i].push_back(cv::Point(i, peak_points[peak_points.size() / 2]));
                peak_points.clear();
            }
        }
        if (peak_points.size() > 0) 
            // peak_points.clear();
            t[i].push_back(cv::Point(i, peak_points[peak_points.size() / 2]));
    }
    // std::sort(vec.begin(), vec.end(), greater_than_pt());
    // std::cout << vec.size() << std::endl;

    // uint j = 0;
    // int  l = 0;
    // int  max_points = 1000;
    // while (l < max_points && j < vec.size()) {
    //     t[vec[j].a].push_back(cv::Point(vec[j].a, vec[j].r));
    //     l++;
    //     j++;
    // }

    int       size = 0;
    for (uint i    = 0; i < t.size(); ++i) {
        size += t[i].size();
    }
    feat_in_polar        = Eigen::MatrixXd::Ones(3, size);
    int       k = 0;
    for (uint i = 0; i < t.size(); ++i) {
        for (uint j = 0; j < t[i].size(); ++j) {
            feat_in_polar(0, k) = t[i][j].x;
            feat_in_polar(1, k) = t[i][j].y;
            k++;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> e = t2 - t1;
    return e.count();
}