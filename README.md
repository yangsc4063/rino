<p align="center">
  <h1 align="center">RINO: Accurate, Robust Radar-Inertial Odometry with Non-Iterative Estimation</h1>

  <p align="center">
    <strong>Shuocheng Yang</strong>
    ·
    <strong>Yueming Cao</strong>
    ·
    <strong>Shengbo Eben Li</strong>
    .
    <strong>jianQiang Wang*</strong>
    .
    <strong>Shaobing Xu*</strong>
</p>

## News

- **[2023/2/28]** 🚀RINO is accepted to T-ASE 2025!
- **[2024/11/12]** Paper released on [arXiv](https://arxiv.org/abs/2411.07699).

## Abstract

Odometry in adverse weather conditions, such as fog, rain, and snow, presents significant challenges, as traditional vision- and LiDAR-based methods often suffer from degraded performance. Radar-Inertial Odometry (RIO) has emerged as a promising solution due to its resilience in such environments. In this paper, we present RINO, a non-iterative RIO framework implemented in an adaptively loosely coupled manner. Building upon ORORA as the baseline for radar odometry, RINO introduces several key advancements, including improvements in keypoint extraction, motion distortion compensation, and pose estimation via an adaptive voting mechanism. This voting strategy facilitates efficient polynomial-time optimization while simultaneously quantifying the uncertainty in the radar module’s pose estimation. The estimated uncertainty is subsequently integrated into the maximum a posteriori (MAP) estimation within a Kalman filter framework. Unlike prior loosely coupled odometry systems, RINO not only retains the global and robust registration capabilities of the radar component but also dynamically accounts for the real-time operational state of each sensor during fusion. Experimental results conducted on publicly available datasets demonstrate that RINO reduces translation and rotation errors by 1.06% and 0.09°/100m, respectively, when compared to the baseline method, thus significantly enhancing its accuracy. Furthermore, RINO achieves performance comparable to state-of-the-art methods.

## Overview

![overview](https://github.com/yangsc4063/rino/blob/main/figure/overview.png)

## Getting Start

### Step-by-step installation instructions

#### 1.Install Dependencies

This project is developed based on ROS Noetic and is recommended to be compiled on Ubuntu 20.04. Please follow the steps below to install the necessary dependencies:

```bash
sudo apt-get install libomp-dev
sudo apt-get install libopencv-dev
sudo apt-get install libeigen3-dev
sudo apt-get install libboost-dev libboost-all-dev
sudo apt-get install libpcl-dev
```

**Sophus**: a C++ library for Lie groups and Lie algebra operations. You need to download and build it manually from the official website:
- Visit the official repository: https://github.com/strasdat/Sophus. Follow the installation instructions provided on the repository to compile and install Sophus.

#### 2.Build RINO

```bash
mkdir ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/yangsc4063/rino.git
cd ..
catkin_make
```

### Prepare Dataset

RINO is developed based on ROS and can read messages published by ROS nodes. 

To use RINO, simply publish your radar and IMU data as ROS topics.

We recommend using the **File Player for MulRan Dataset** to simulate the input data. The MulRan dataset provides both radar and IMU data suitable for RINO’s input requirements.

- You can find the **File Player for MulRan Dataset** here: https://github.com/RPM-Robotics-Lab/file_player_mulran.

### Run

```bash
cd ~/catkin_ws
source devel/setup.bash
roslaunch rino rino.launch
```

## Primary Results

**Result on <u>KAIST02</u> of MulRan dataset:**

<div align="center">
    <img src=".\figure\result_kaist.png" alt="result_kaist" width="60%" />
</div>

**Result on <u>2021-11-28-09-18</u> of Boreas dataset:**

https://github.com/user-attachments/assets/7aafe2a6-03fb-4cba-80e3-fece956c6a8e

## Acknowledgement

Many thanks to these excellent projects:

- [ORORA](https://github.com/url-kaist/outlier-robust-radar-odometry)

- [SLAM in autonomous driving](https://github.com/gaoxiang12/slam_in_autonomous_driving)

