//
// Created by Yangsc on 23-11-10.
//

#ifndef POSE_SOLVER_HPP
#define POSE_SOLVER_HPP

#include <unistd.h>
#include <iostream>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "common/conversion.hpp"
#include "common/eigen_types.hpp"
#include "common/radar_utils.hpp"
#include "odom/graph.hpp"

class Pose_Solver {
public:
    enum class INLIER_SELECTION_MODE { // 内点（inlier）选择模式
        PMC_EXACT = 0, // 严格的PMC
        PMC_HEU   = 1, // 启发式的PMC
        KCORE_HEU = 2, // 启发式的K-Core
        NONE      = 3, // 不进行内点选择
    };

    /** \brief Empty constructor. */
    Pose_Solver() {};
    /** \brief Empty destructor */
    ~Pose_Solver() {};

    inline void reset() {
        inliers_.clear();
        inlier_graph_.clear();
    }

    inline void setInputSource(const Eigen::MatrixXd &cloudEigen) {
        int N = cloudEigen.cols();
        N_ = N;
        src_matched_.resize(2, N);
        src_matched_.topRows(2)    = cloudEigen.topRows(2);
    }

    inline void setInputTarget(const Eigen::MatrixXd &cloudEigen) {
        int N = cloudEigen.cols();
        assert(N == N_);
        tgt_matched_.resize(2, N);
        tgt_matched_.topRows(2)    = cloudEigen.topRows(2);
    }

    inline void setInputSrcFeat(const Eigen::MatrixXd &cloudEigen) {
        int N = cloudEigen.cols();
        src_feat_.resize(2, N);
        src_feat_.topRows(2)    = cloudEigen.topRows(2);
    }

    inline void setInputTgtFeat(const Eigen::MatrixXd &cloudEigen) {
        int N = cloudEigen.cols();
        tgt_feat_.resize(2, N);
        tgt_feat_.topRows(2)    = cloudEigen.topRows(2);
    }

    Eigen::Matrix2Xd computeTIMs(const Eigen::Matrix2Xd &matched,
                                 Eigen::Matrix2Xi &tims_map,
                                 int num);
    
    Eigen::MatrixXd computeHardSC(const Eigen::VectorXd &src_dists,
                                  const Eigen::VectorXd &tgt_dists,
                                  double inlier_thr);
    
    std::vector<int> selectInliers(const Eigen::MatrixXd &sc_);

    // std::vector<Mat3d> estimateUncertainty(const Eigen::Matrix2Xd &pruned_tgt);

    void solveForRotation(const Eigen::Matrix2Xd &src_tims,
                          const Eigen::Matrix2Xd &tgt_tims);

    void solveForTranslation(const Eigen::Matrix2Xd &pruned_src,
                             const Eigen::Matrix2Xd &pruned_tgt);

    void computeTransformation(const SE3 &inital_guess, SE3 &output, Vec3d &uncertainty, int &inlier);

private:
    Mat2d est_rot_    = Mat2d::Identity();
    double ucer_rot_;
    Vec2d est_trans_  = Vec2d::Zero();
    Vec2d ucer_trans_ = Vec2d::Zero();

    Graph inlier_graph_;

    Eigen::Matrix2Xd src_feat_;
    Eigen::Matrix2Xd tgt_feat_;

    // points
    Eigen::Matrix2Xd src_matched_;
    Eigen::Matrix2Xd tgt_matched_;
    // points after pruning
    Eigen::Matrix2Xd pruned_src_;
    Eigen::Matrix2Xd pruned_tgt_;

    // TIMs
    Eigen::Matrix2Xd src_tims_;
    Eigen::Matrix2Xd tgt_tims_;
    // TIM maps
    Eigen::Matrix2Xi src_tims_map_;
    Eigen::Matrix2Xi tgt_tims_map_;
    // TIMs after pruning
    Eigen::Matrix2Xd pruned_src_tims_;
    Eigen::Matrix2Xd pruned_tgt_tims_;

    // Cov of each inlier point
    Eigen::Matrix4Xd src_cov_;
    Eigen::Matrix4Xd tgt_cov_;
    // Cov of each inlier tims
    Eigen::Matrix4Xd src_tims_cov_;
    Eigen::Matrix4Xd tgt_tims_cov_;

    // dists
    Eigen::VectorXd src_dists_;
    Eigen::VectorXd tgt_dists_;

    // SC
    Eigen::MatrixXd hard_sc_;

    // inliers
    std::vector<int> inliers_;

private:
    // Parameters for 
    int N_;
    int M_;
    // Parameters for uncertainty estimation
    double cbar_radial_ = 0.1;
    double cbar_tangential_ = 1.5 / 180 * M_PI;
    // double cbar_tangential_ = 10.8;
    // Parameters for gnc
    double cbar_rot_ = 0.6;
    double rotation_gnc_factor_ = 1.4;
    size_t rotation_max_iterations_ = 100;
    double rotation_cost_threshold_ = 1e-6;

    // 内点（inlier）选择模式
    // INLIER_SELECTION_MODE inlier_selection_mode = NonMinimalSolver::INLIER_SELECTION_MODE::PMC_HEU;
    INLIER_SELECTION_MODE inlier_selection_mode_ = INLIER_SELECTION_MODE::PMC_EXACT;
    /**
     * \brief 用于确定是否跳过最大团选择而直接进行 GNC 旋转估计的阈值比率。
     * 将其设置为1时，始终使用精确的最大团选择；将其设置为0时，始终跳过精确的最大团选择。
     */
    double kcore_heuristic_threshold_ = 0.5;
    bool use_max_clique_ = true;            // 是否使用最大团选择 
    bool max_clique_exact_solution_ = true; // 是否使用精确的最大团选择
    double max_clique_time_limit_ = 3600;   // 最大团选择的时间限制

    const double inlier_thr_ = 0.3;
};

#endif // POSE_SOLVER_HPP