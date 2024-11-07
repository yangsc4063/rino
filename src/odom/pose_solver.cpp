//
// Created by Yangsc on 23-11-10.
// The code is based on ORORA.
//

#include <algorithm>
#include <execution>

#include "odom/pose_solver.hpp"

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>


Eigen::Matrix2Xd Pose_Solver::computeTIMs(const Eigen::Matrix2Xd &matched,
                                          Eigen::Matrix2Xi &tims_map, 
                                          int num) {
    
    Eigen::Matrix2Xd tims(2, num * (num - 1) / 2);
    tims_map.resize(2, num * (num - 1) / 2);
    
#pragma omp parallel for default(none) shared(num, matched, tims, tims_map)
    for (size_t i = 0; i < num - 1; i++) {
        /**
         * 计算每个段的起始索引
         * 例如：i=0：添加N-1个TIMs；i=1：添加N-2个TIMs；依此类推；i=k：添加N-1-k个TIMs
         * 通过等差数列，我们可以得到每个段的起始索引为：k*N - k(k+1)/2
         */
        size_t segment_start_idx = i * num - i * (i + 1) / 2;
        size_t segment_cols      = num - 1 - i;
        
        // calculate TIM
        Vec2d pi = matched.col(i);
        Eigen::Matrix2Xd temp = matched - pi * Eigen::MatrixXd::Ones(1, num);
        tims.middleCols(segment_start_idx, segment_cols) = temp.rightCols(segment_cols);

        // populate the index map
        Eigen::Matrix2Xi temp_map(2, num); // 用于存储 TIMs 的索引
        for (size_t j = 0; j < num; ++j) {
            temp_map(0, j) = i;
            temp_map(1, j) = j;
        }

        tims_map.middleCols(segment_start_idx, segment_cols) = temp_map.rightCols(segment_cols);
    }

    return tims;
}

Eigen::MatrixXd Pose_Solver::computeHardSC(const Eigen::VectorXd &src_dists,
                                           const Eigen::VectorXd &tgt_dists,
                                           double inlier_thr) {
    
    Eigen::MatrixXd hard_sc = Eigen::MatrixXd::Zero(N_, N_);

    Eigen::VectorXd dist = src_dists - tgt_dists;
#pragma omp parallel for
    for (size_t i = 0; i < src_dists.rows(); ++i) {
        if (std::abs(dist(i)) < inlier_thr) {
            hard_sc(src_tims_map_(0, i), src_tims_map_(1, i)) = 1;
            hard_sc(src_tims_map_(1, i), src_tims_map_(0, i)) = 1;
        }
    }
    // std::cout << "Hard SC: " << std::endl << hard_sc << std::endl;
    
    // cv::Mat hard_sc_img;
    // cv::eigen2cv(hard_sc, hard_sc_img);
    // cv::imshow("Test for hard_sc", hard_sc_img);
    // cv::waitKey(20);

    return hard_sc;
}

std::vector<int> Pose_Solver::selectInliers(const Eigen::MatrixXd &sc_) {
    
    std::vector<int> inliers;

    if (inlier_selection_mode_ != INLIER_SELECTION_MODE::NONE) {

        inlier_graph_.populateVertices(sc_.cols());
        for (size_t i = 0; i < sc_.rows(); ++i) {
            for (size_t j = i; j < sc_.cols(); ++j) {    
                if (sc_(i, j) != 0) {
                    inlier_graph_.addEdge(i, j);
                }
            }
        }

        MaxCliqueSolver::Params clique_params;
        if (inlier_selection_mode_ == INLIER_SELECTION_MODE::PMC_EXACT) {
            clique_params.solver_mode = MaxCliqueSolver::CLIQUE_SOLVER_MODE::PMC_EXACT;
        } 
        else if (inlier_selection_mode_ == INLIER_SELECTION_MODE::PMC_HEU) {
            clique_params.solver_mode = MaxCliqueSolver::CLIQUE_SOLVER_MODE::PMC_HEU;
        } 
        else {
            clique_params.solver_mode = MaxCliqueSolver::CLIQUE_SOLVER_MODE::KCORE_HEU;
        }
        clique_params.time_limit                = max_clique_time_limit_;
        clique_params.kcore_heuristic_threshold = kcore_heuristic_threshold_;
        
        MaxCliqueSolver clique_solver(clique_params);
        inliers = clique_solver.findMaxClique(inlier_graph_);

        std::sort(inliers.begin(), inliers.end());

        if (inliers.size() <= 1) {
            std::cout << "\033[1;31m[Warning] Too few " << inliers.size() << " cliques exist!!!\033[0m" << std::endl;
            std::cin.ignore();
            return inliers;
        }
    }

    return inliers;
}

inline double calcCov11(double c, double s, double rho, double var_r, double var_t) {
    return c * c * var_r + s * s * rho * rho * var_t;
}
inline double calcCov12(double c, double s, double rho, double var_r, double var_t) {
    return c * s * var_r - c * s * rho * rho * var_t;
}
inline double calcCov21(double c, double s, double rho, double var_r, double var_t) {
    return c * s * var_r + c * s * rho * rho * var_t;
}
inline double calcCov22(double c, double s, double rho, double var_r, double var_t) {
    return s * s * var_r + c * c * rho * rho * var_t;
}

void estimate(const Eigen::RowVectorXd &X,
              const Eigen::RowVectorXd &ranges,
              double *estimate,
              double *uncertainty) {
    
    // check input parameters
    bool dimension_inconsistent = (X.rows() != ranges.rows()) || (X.cols() != ranges.cols());
    bool only_one_element = (X.rows() == 1) && (X.cols() == 1);
    assert(!dimension_inconsistent);
    assert(!only_one_element); // TODO: admit a trivial solution

    int                                 N = X.cols();
    std::vector<std::pair<double, int>> h;
    for (size_t                         i = 0; i < N; ++i) {
        h.push_back(std::make_pair(X(i) - ranges(i), i + 1));
        h.push_back(std::make_pair(X(i) + ranges(i), -i - 1));
    }

    // ascending order
    std::sort(h.begin(), h.end(), [](std::pair<double, int> a, std::pair<double, int> b) { return a.first < b.first; });

    // calculate weights
    Eigen::RowVectorXd weights = ranges.array().square();
    weights = weights.array().inverse();
    int                nr_centers      = 2 * N;
    Eigen::RowVectorXd x_hat           = Eigen::MatrixXd::Zero(1, nr_centers);
    Eigen::RowVectorXd sigma_hat       = Eigen::MatrixXd::Zero(1, nr_centers);
    Eigen::RowVectorXd x_cost          = Eigen::MatrixXd::Zero(1, nr_centers);

    double ranges_inverse_sum     = ranges.sum();
    double dot_X_weights          = 0;
    double dot_weights_consensus  = 0;
    int    consensus_set_cardinal = 0;
    double sum_xi                 = 0;
    double sum_xi_square          = 0;

    for (size_t i = 0; i < nr_centers; ++i) {
        int idx     = int(std::abs(h.at(i).second)) - 1;
        int epsilon = (h.at(i).second > 0) ? 1 : -1;

        consensus_set_cardinal += epsilon;
        dot_weights_consensus += epsilon * weights(idx);
        dot_X_weights += epsilon * weights(idx) * X(idx);
        ranges_inverse_sum -= epsilon * ranges(idx);
        sum_xi += epsilon * X(idx);
        sum_xi_square += epsilon * X(idx) * X(idx);

        x_hat(i)     = dot_X_weights / dot_weights_consensus;
        sigma_hat(i) = 1/ dot_weights_consensus;
        // sum_xi_square: already includes consensus_set_cardinal
        double residual = consensus_set_cardinal * x_hat(i) * x_hat(i) + sum_xi_square - 2 * sum_xi * x_hat(i);
        x_cost(i) = residual + ranges_inverse_sum;
    }

    size_t min_idx;
    x_cost.minCoeff(&min_idx);

    double estimate_temp = x_hat(min_idx);
    double uncert_temp = sigma_hat(min_idx);

    if (estimate) {
        // update estimate output if it's not nullptr
        *estimate    = estimate_temp;
        *uncertainty = uncert_temp;
    }
}

void estimate_tiled(const Eigen::RowVectorXd& X,
                    const Eigen::RowVectorXd& ranges, const int& s,
                    double* estimate) {
    // check input parameters
    bool dimension_inconsistent = (X.rows() != ranges.rows()) || (X.cols() != ranges.cols());
    bool only_one_element = (X.rows() == 1) && (X.cols() == 1);
    assert(!dimension_inconsistent);
    assert(!only_one_element); // TODO: admit a trivial solution

    // Prepare variables for calculations
    int N = X.cols();
    Eigen::RowVectorXd h(N * 2);
    h << X - ranges, X + ranges;
    // ascending order
    std::sort(h.data(), h.data() + h.cols(), [](double a, double b) { return a < b; });
    // calculate interval centers
    Eigen::RowVectorXd h_centers = (h.head(h.cols() - 1) + h.tail(h.cols() - 1)) / 2;
    auto nr_centers = h_centers.cols();

    // calculate weights
    Eigen::RowVectorXd weights = ranges.array().square();
    weights = weights.array().inverse();

    Eigen::RowVectorXd x_hat = Eigen::MatrixXd::Zero(1, nr_centers);
    Eigen::RowVectorXd x_cost = Eigen::MatrixXd::Zero(1, nr_centers);

    // loop tiling
    size_t ih_bound = ((nr_centers) & ~((s)-1));
    size_t jh_bound = ((N) & ~((s)-1));

    std::vector<double> ranges_inverse_sum_vec(nr_centers, 0);
    std::vector<double> dot_X_weights_vec(nr_centers, 0);
    std::vector<double> dot_weights_consensus_vec(nr_centers, 0);
    std::vector<std::vector<double>> X_consensus_table(nr_centers, std::vector<double>());

    auto inner_loop_f = [&](const size_t& i, const size_t& jh, const size_t& jl_lower_bound,
                            const size_t& jl_upper_bound) {
        double& ranges_inverse_sum = ranges_inverse_sum_vec[i];
        double& dot_X_weights = dot_X_weights_vec[i];
        double& dot_weights_consensus = dot_weights_consensus_vec[i];
        std::vector<double>& X_consensus_vec = X_consensus_table[i];

        size_t j = 0;
        for (size_t jl = jl_lower_bound; jl < jl_upper_bound; ++jl) {
            j = jh + jl;
            bool consensus = std::abs(X(j) - h_centers(i)) <= ranges(j);
            if (consensus) {
                dot_X_weights += X(j) * weights(j);
                dot_weights_consensus += weights(j);
                X_consensus_vec.push_back(X(j));
            } 
            else {
                ranges_inverse_sum += ranges(j);
            }
        }

        if (j == N - 1) {
            // x_hat(i) = dot(X(consensus), weights(consensus)) / dot(weights, consensus);
            x_hat(i) = dot_X_weights / dot_weights_consensus;

            // residual = X(consensus)-x_hat(i);
            Eigen::Map<Eigen::VectorXd> X_consensus(X_consensus_vec.data(), X_consensus_vec.size());
            Eigen::VectorXd residual = X_consensus.array() - x_hat(i);

            // x_cost(i) = dot(residual,residual) + sum(ranges(~consensus));
            x_cost(i) = residual.squaredNorm() + ranges_inverse_sum;
        }
    };

#pragma omp parallel for default(none) shared(                                                     \
        jh_bound, ih_bound, ranges_inverse_sum_vec, dot_X_weights_vec, dot_weights_consensus_vec,  \
        X_consensus_table, h_centers, weights, N, X, x_hat, x_cost, s, ranges, inner_loop_f)
    for (size_t ih = 0; ih < ih_bound; ih += s) {
        for (size_t jh = 0; jh < jh_bound; jh += s) {
            for (size_t il = 0; il < s; ++il) {
                size_t i = ih + il;
                inner_loop_f(i, jh, 0, s);
            }
        }
    }

  // finish the left over entries
  // 1. Finish the unfinished js
#pragma omp parallel for default(none) shared(                                                     \
           jh_bound, ih_bound, ranges_inverse_sum_vec, dot_X_weights_vec,                          \
           dot_weights_consensus_vec, X_consensus_table, h_centers, weights, N, X, x_hat, x_cost,  \
           s, ranges, nr_centers, inner_loop_f)
    for (size_t i = 0; i < nr_centers; ++i) {
        inner_loop_f(i, 0, jh_bound, N);
    }

  // 2. Finish the unfinished is
#pragma omp parallel for default(none) shared(                                                     \
           jh_bound, ih_bound, ranges_inverse_sum_vec, dot_X_weights_vec,                          \
           dot_weights_consensus_vec, X_consensus_table, h_centers, weights, N, X, x_hat, x_cost,  \
           s, ranges, nr_centers, inner_loop_f)
    for (size_t i = ih_bound; i < nr_centers; ++i) {
        inner_loop_f(i, 0, 0, N);
    }

    size_t min_idx;
    x_cost.minCoeff(&min_idx);
    double estimate_temp = x_hat(min_idx);
    if (estimate) {
        // update estimate output if it's not nullptr
        *estimate = estimate_temp;
    }
}

inline Mat3d svdRot(const Eigen::Matrix<double, 3, Eigen::Dynamic> &X,
                    const Eigen::Matrix<double, 3, Eigen::Dynamic> &Y,
                    const Eigen::Matrix<double, 1, Eigen::Dynamic> &W) {
    // Assemble the correlation matrix H = X * Y'
    Mat3d H = X * W.asDiagonal() * Y.transpose();

    Eigen::JacobiSVD<Mat3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3d                   U = svd.matrixU();
    Mat3d                   V = svd.matrixV();

    if (U.determinant() * V.determinant() < 0) {
        V.col(1) *= -1;
    }

    return V * U.transpose();
}

void Pose_Solver::solveForRotation(const Eigen::Matrix2Xd &src_tims,
                                   const Eigen::Matrix2Xd &tgt_tims) {
    // 1.Calculate raw rotation & variances
    // raw_rotation
    Eigen::RowVectorXd raw_rotation(M_);
    // alpha
    Eigen::RowVectorXd alphas(M_);

#pragma omp parallel for 
    for (size_t i = 0; i < M_; ++i) {
        double src_tim_ang = atan2(src_tims(1, i), src_tims(0, i));
        double tgt_tim_ang = atan2(tgt_tims(1, i), tgt_tims(0, i));

        raw_rotation(i) = tgt_tim_ang - src_tim_ang;

        double src_tim_rho = src_tims.col(i).norm();
        double tgt_tim_rho = tgt_tims.col(i).norm();
        double src_tim_c   = cos(M_PI/2 - src_tim_ang);
        double src_tim_s   = sin(M_PI/2 - src_tim_ang);
        double tgt_tim_c   = cos(M_PI/2 - tgt_tim_ang);
        double tgt_tim_s   = sin(M_PI/2 - tgt_tim_ang);

        Eigen::VectorXd src_tim_cov = src_tims_cov_.col(i);
        Eigen::VectorXd tgt_tim_cov = tgt_tims_cov_.col(i);

        double src_sigma_sq = src_tim_c * src_tim_c * src_tim_cov(0) - 
                              src_tim_c * src_tim_s * src_tim_cov(1) -
                              src_tim_s * src_tim_c * src_tim_cov(2) +
                              src_tim_s * src_tim_s * src_tim_cov(3);
        double tgt_sigma_sq = tgt_tim_c * tgt_tim_c * tgt_tim_cov(0) -
                              tgt_tim_c * tgt_tim_s * tgt_tim_cov(1) -
                              tgt_tim_s * tgt_tim_c * tgt_tim_cov(2) +
                              tgt_tim_s * tgt_tim_s * tgt_tim_cov(3);

        alphas(i) = sqrt(atan2(sqrt(src_sigma_sq), src_tim_rho) * atan2(sqrt(src_sigma_sq), src_tim_rho) + 
                         atan2(sqrt(tgt_sigma_sq), tgt_tim_rho) * atan2(sqrt(tgt_sigma_sq), tgt_tim_rho));
    }

    // 2.COTE
    double est_rot_ang;
    estimate(raw_rotation, alphas, &est_rot_ang, &ucer_rot_);
    // estimate_tiled(raw_rotation, alphas, 2, &est_rot_ang);
    est_rot_ << cos(est_rot_ang), -sin(est_rot_ang),
                sin(est_rot_ang),  cos(est_rot_ang);

    // // 1.GNC
    // assert(src_tims.cols() == tgt_tims.cols()); // check dimensions of input data
    // assert(rotation_gnc_factor_ > 1);     // make sure mu will increase
    // assert(cbar_rot_ != 0);               // make sure cbar is not zero
    
    // Mat3d rotation = Mat3d::Identity();

    // // Prepare some variables
    // size_t N = src_tims.cols(); // number of correspondences
    // double mu = 1; // arbitrary starting mu
    // double prev_cost = std::numeric_limits<double>::infinity();
    // double cost = std::numeric_limits<double>::infinity();
    // double cbar_sq = std::pow(cbar_rot_, 2);
    // if (cbar_sq < 1e-16) {
    //     cbar_sq = 1e-2;
    // }

    // Eigen::Matrix<double, 3, Eigen::Dynamic> diffs(3, N);
    // Eigen::Matrix<double, 1, Eigen::Dynamic> weights(1, N);
    // weights.setOnes(1, N);
    // Eigen::Matrix<double, 1, Eigen::Dynamic> residuals_sq(1, N);
    // Eigen::Matrix3Xd src_tims_3d(3, N);
    // src_tims_3d.block<2, Eigen::Dynamic>(0, 0, 2, N) = src_tims;
    // src_tims_3d.row(2).setZero();
    // Eigen::Matrix3Xd tgt_tims_3d(3, N);
    // tgt_tims_3d.block<2, Eigen::Dynamic>(0, 0, 2, N) = tgt_tims;
    // tgt_tims_3d.row(2).setZero();

    // // Calculating Initial weights
    // // Calculate residuals squared
    // rotation     = svdRot(src_tims_3d, tgt_tims_3d, weights);
    // diffs        = (tgt_tims_3d - rotation * src_tims_3d).array().square();
    // residuals_sq = diffs.colwise().sum();

    // static double EPSILON = 0.00000000000001;

    // double        max_residual = residuals_sq.maxCoeff();
    // if (max_residual < cbar_sq) {
    //     cbar_sq = 0.01; // Forcely set noise bound into 0.1 => 0.1 * 0.1
    // }
    // mu = 1 / (2 * max_residual / cbar_sq - 1);
    // // Degenerate case: mu = -1 because max_residual is very small
    // // i.e., little to none noise
    // if (mu <= 0) {
    //     est_rot_ = Mat2d::Identity();
    //     return;
    // }

    // // Fix R and solve for weights in closed form
    // double th1 = (mu + 1) / mu * cbar_sq;
    // double th2 = mu / (mu + 1) * cbar_sq;
    // cost = 0;

    // for (size_t j = 0; j < N; ++j) {
    //     // Also calculate cost in this loop
    //     // Note: the cost calculated is using the previously solved weights
    //     cost += weights(j) * residuals_sq(j);

    //     if (residuals_sq(j) >= th1) {
    //         weights(j) = 0;
    //     } 
    //     else if (residuals_sq(j) <= th2) {
    //         weights(j) = 1;
    //     } 
    //     else {
    //         weights(j) = sqrt(cbar_sq * mu * (mu + 1) / residuals_sq(j) + EPSILON) - mu;
    //         assert(weights(j) >= 0 && weights(j) <= 1);
    //     }
    // }
    // mu = mu * rotation_gnc_factor_;

    // // Loop for performing GNC-TLS
    // for (size_t i = 0; i < rotation_max_iterations_; ++i) {
    //     // Fix weights and perform SVD rotation estimation
    //     rotation     = svdRot(src_tims_3d, tgt_tims_3d, weights);
    //     diffs        = (tgt_tims_3d - rotation * src_tims_3d).array().square();
    //     residuals_sq = diffs.colwise().sum();
    //     // Fix R and solve for weights in closed form
    //     double th1 = (mu + 1) / mu * cbar_sq;
    //     double th2 = mu / (mu + 1) * cbar_sq;
    //     cost = 0;

    //     for (size_t j = 0; j < N; ++j) {
    //         // Also calculate cost in this loop
    //         // Note: the cost calculated is using the previously solved weights
    //         cost += weights(j) * residuals_sq(j);
    //         if (residuals_sq(j) >= th1) {
    //             weights(j) = 0;
    //         } 
    //         else if (residuals_sq(j) <= th2) {
    //             weights(j) = 1;
    //         } 
    //         else {
    //             weights(j) = sqrt(cbar_sq * mu * (mu + 1) / residuals_sq(j) + EPSILON) - mu;
    //             assert(weights(j) >= 0 && weights(j) <= 1);
    //         }
    //     }
    //     // Calculate cost
    //     double cost_diff = std::abs(cost - prev_cost);
    //     // Increase mu
    //     mu = mu * rotation_gnc_factor_;
    //     prev_cost = cost;
        
    //     if (cost_diff < rotation_cost_threshold_) {
    //         std::cout << "\e[34m[Tips]GNC-TLS solver terminated due to cost convergence.\e[0m" << std::endl;
    //         std::cout << "\e[34m[Tips]Cost diff: \e[0m" << cost_diff << std::endl;
    //         std::cout << "\e[34m[Tips]Iterations: \e[0m" << i << std::endl;
    //         // std::cout << "\e[34mmu: \e[0m" << mu << std::endl;
    //         break;
    //     }
    // }

    // est_rot_ = rotation.block<2, 2>(0, 0);
}

void Pose_Solver::solveForTranslation(const Eigen::Matrix2Xd &pruned_src,
                                      const Eigen::Matrix2Xd &pruned_tgt) {
    // 1.Calculate raw translation & variances
    // raw translation
    Eigen::Matrix2Xd raw_translation(2, M_);
    // alpha
    Eigen::Matrix2Xd alphas(2, M_);

    raw_translation = pruned_tgt - est_rot_ * pruned_src;

    for (size_t i = 0; i < M_; ++i) {
        Eigen::VectorXd src_tim_cov = src_tims_cov_.col(i);
        Eigen::VectorXd tgt_tim_cov = tgt_tims_cov_.col(i);

        alphas(0, i) = sqrt(tgt_tim_cov(0) + 
                           (est_rot_(0, 0) * src_tim_cov(0) * est_rot_(0, 0) + 
                            est_rot_(0, 1) * src_tim_cov(1) * est_rot_(0, 0) + 
                            est_rot_(0, 0) * src_tim_cov(2) * est_rot_(0, 1) + 
                            est_rot_(0, 1) * src_tim_cov(3) * est_rot_(0, 1)));
        alphas(1, i) = sqrt(tgt_tim_cov(3) + 
                           (est_rot_(1, 0) * src_tim_cov(0) * est_rot_(1, 0) + 
                            est_rot_(1, 1) * src_tim_cov(1) * est_rot_(1, 0) + 
                            est_rot_(1, 0) * src_tim_cov(2) * est_rot_(1, 1) + 
                            est_rot_(1, 1) * src_tim_cov(3) * est_rot_(1, 1)));
    }

    // 2.ACOTE
    estimate(raw_translation.row(0), alphas.row(0), &((est_trans_)(0)), &((ucer_trans_)(0)));
    estimate(raw_translation.row(1), alphas.row(1), &((est_trans_)(1)), &((ucer_trans_)(1)));
    // estimate_tiled(raw_translation.row(0), alphas.row(0), 2, &((est_trans_)(0)));
    // estimate_tiled(raw_translation.row(1), alphas.row(1), 2, &((est_trans_)(1)));
}

void Pose_Solver::computeTransformation(const SE3 &inital_guess, SE3 &output, Vec3d &uncertainty, int &inlier) {
    // Once estimation fails, it returns identity se3
    output = SE3(Mat3d::Identity(), Vec3d::Zero());
    uncertainty = Vec3d::Zero();

    std::cout << "\033[1;32m[Tips]The number of correspondences: \033[0m" << N_ << std::endl;

    // cv::Mat back_ground(964, 964, CV_8UC1, cv::Scalar(0));
    // draw_points(back_ground, 
    //             src_matched_, tgt_matched_, 
    //             0.2592, 964, 
    //             back_ground, 
    //             {255, 0, 0}, {0, 0, 255});
    // cv::imshow("Test for matching", back_ground);
    // cv::waitKey(20);
    
    // 1.Generate TIMs
    src_tims_ = computeTIMs(src_matched_, src_tims_map_, N_);
    src_dists_ = src_tims_.colwise().norm();
    tgt_tims_ = computeTIMs(tgt_matched_, tgt_tims_map_, N_);
    tgt_dists_ = tgt_tims_.colwise().norm();

    // 2.Find maximum clique
    hard_sc_ = computeHardSC(src_dists_, tgt_dists_, inlier_thr_);
    inliers_ = selectInliers(hard_sc_);
    std::cout << "\033[1;32m[Tips]The number of inliers: \033[0m" << inliers_.size() << std::endl;

    pruned_src_.resize(2, inliers_.size());
    pruned_tgt_.resize(2, inliers_.size());
    for (size_t i = 0; i < inliers_.size(); ++i) {
        pruned_src_.col(i) = src_matched_.col(inliers_[i]);
        pruned_tgt_.col(i) = tgt_matched_.col(inliers_[i]);
    }
    M_ = inliers_.size();

    // cv::Mat back_ground(964, 964, CV_8UC1, cv::Scalar(0));
    // draw_points(back_ground, 
    //             pruned_src_, pruned_tgt_, 
    //             0.2592, 964, 
    //             back_ground, 
    //             {255, 0, 0}, {0, 0, 255});
    // cv::imshow("Test for inliers' matching", back_ground);
    // cv::waitKey(20);

    // cv::Mat back_ground(964, 964, CV_8UC1, cv::Scalar(0));
    // draw_points(back_ground, 
    //             src_feat_, pruned_src_, 
    //             0.2592, 964, 
    //             back_ground, 
    //             {255, 0, 0}, {0, 0, 255});
    // cv::imshow("Test for keypoint", back_ground);
    // cv::waitKey(20);

    // 3.Evaluate the measurement uncertainty at each inlier point
    // Prepare some variables
    src_cov_.resize(4, M_);
    tgt_cov_.resize(4, M_);

    double var_r = cbar_radial_ * cbar_radial_;
    double var_t = cbar_tangential_ * cbar_tangential_;

    static double EPSILON = 0.00000000000001;

    for (size_t i = 0; i < M_; ++i) {
        double src_rho = pruned_src_.col(i).norm();
        double tgt_rho = pruned_tgt_.col(i).norm();
        double src_ang = atan2(pruned_src_(1, i), pruned_src_(0, i));
        double tgt_ang = atan2(pruned_tgt_(1, i), pruned_tgt_(0, i));
        double src_c   = cos(src_ang);
        double src_s   = sin(src_ang);
        double tgt_c   = cos(tgt_ang);
        double tgt_s   = sin(tgt_ang);

        double src_cov_11 = calcCov11(src_c, src_s, src_rho, var_r, var_t);
        double src_cov_21 = calcCov21(src_c, src_s, src_rho, var_r, var_t);
        double src_cov_12 = calcCov12(src_c, src_s, src_rho, var_r, var_t);
        double src_cov_22 = calcCov22(src_c, src_s, src_rho, var_r, var_t);
        double tgt_cov_11 = calcCov11(tgt_c, tgt_s, tgt_rho, var_r, var_t);
        double tgt_cov_21 = calcCov21(tgt_c, tgt_s, tgt_rho, var_r, var_t);
        double tgt_cov_12 = calcCov12(tgt_c, tgt_s, tgt_rho, var_r, var_t);
        double tgt_cov_22 = calcCov22(tgt_c, tgt_s, tgt_rho, var_r, var_t);

        src_cov_.col(i) << src_cov_11, src_cov_21, src_cov_12, src_cov_22;
        tgt_cov_.col(i) << tgt_cov_11, tgt_cov_21, tgt_cov_12, tgt_cov_22;
    }

    // 4.Solve for rotation
    // 4.1.Calculate new TIMs based on max clique inliers
    pruned_src_tims_.resize(2, M_);
    pruned_tgt_tims_.resize(2, M_);
    src_tims_cov_.resize(4, M_);
    tgt_tims_cov_.resize(4, M_);

    // Inlier graph is represented as a chain graph
    for (size_t i = 0; i < M_; ++i) {
        if (i != M_ - 1) {
            pruned_src_tims_.col(i) = pruned_src_.col(i + 1) - pruned_src_.col(i);
            pruned_tgt_tims_.col(i) = pruned_tgt_.col(i + 1) - pruned_tgt_.col(i);
            // evaluate the measurement uncertainty at each inlier tims
            src_tims_cov_.col(i) = src_cov_.col(i + 1) + src_cov_.col(i);
            tgt_tims_cov_.col(i) = tgt_cov_.col(i + 1) + tgt_cov_.col(i);
        }
        else {
            pruned_src_tims_.col(i) = pruned_src_.col(0) - pruned_src_.col(i);
            pruned_tgt_tims_.col(i) = pruned_tgt_.col(0) - pruned_tgt_.col(i);
            // evaluate the measurement uncertainty at each inlier tims
            src_tims_cov_.col(i) = src_cov_.col(0) + src_cov_.col(i);
            tgt_tims_cov_.col(i) = tgt_cov_.col(0) + tgt_cov_.col(i);
        }
    }
    
    // cv::Mat back_ground(964, 964, CV_8UC1, cv::Scalar(0));
    // draw_arrows(back_ground, 
    //             pruned_src_, pruned_tgt_, 
    //             0.2592, 964, 
    //             back_ground, 
    //             {255, 0, 0}, {0, 0, 255});
    // cv::imshow("Test for pruned tims", back_ground);
    // cv::waitKey(20);

    // 4.2.Solve for rotation
    solveForRotation(pruned_src_tims_, pruned_tgt_tims_);

    // 5.Solve for translation
    solveForTranslation(pruned_src_, pruned_tgt_);

    Mat3d Rotation = Mat3d::Identity();
    Rotation.block<2, 2>(0, 0) = est_rot_;

    Vec3d Translation = Vec3d::Zero();
    Translation.head<2>() = est_trans_;
    
    output = SE3(Rotation, Translation);
    uncertainty << ucer_rot_, ucer_trans_(0), ucer_trans_(1);
    inlier = inliers_.size();
}