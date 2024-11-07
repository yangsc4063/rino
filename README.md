<p align="center">
  <h1 align="center">RINO: Accurate, Robust Radar-Inertial Odometry with Non-Iterative Estimation</h1>

  <p align="center">
    <strong>Shuocheng Yang</strong>
    ·
    <strong>Yueming Cao</strong>
    ·
    <strong>Shengbo Li</strong>
    .
    <strong>jianQiang Wang</strong>
    .
    <strong>Shaobing Xu*</strong>
</p>

## Abstract

Precise localization and mapping are critical for achieving autonomous navigation in self-driving vehicles. However, ego-motion estimation still faces significant challenges, particularly when GNSS failures occur or under extreme weather conditions (e.g., fog, rain, and snow). In recent years, scanning radar has emerged as an effective solution due to its strong penetration capabilities. Nevertheless, scanning radar data inherently contains high levels of noise, necessitating hundreds to thousands of iterations of optimization to estimate a reliable transformation from the noisy data. Such iterative solving is time-consuming, unstable, and prone to failure. To address these challenges, we propose an accurate and robust Radar-Inertial Odometry system, RINO, which employs a non-iterative solving approach. Our method decouples rotation and translation estimation and applies an adaptive voting scheme for 2D rotation estimation, enhancing efficiency while ensuring consistent solving time. Additionally, the approach implements a loosely coupled system between the scanning radar and an inertial measurement unit (IMU), leveraging Error-State Kalman Filtering (ESKF). Notably, we successfully estimated the uncertainty of the pose estimation from the scanning radar, incorporating this into the filter's Maximum A Posteriori estimation, a consideration that has been previously overlooked. Validation on publicly available datasets demonstrates that RINO outperforms state-of-the-art methods and baselines in both accuracy and robustness.

## Getting Start

### Step-by-step installation instructions

#### 1.Install Dependencies

This project is developed based on ROS Noetic and is recommended to be compiled on Ubuntu 20.04. Please follow the steps below to install the necessary dependencies:

- **OpenMP**: 

  ```bash
  sudo apt-get install libomp-dev
  ```

- **OpenCV**:

  ```bash
  sudo apt-get install libopencv-dev
  ```

- **Eigen3**: Eigen3 is a C++ template library for linear algebra operations.

  ```bash
  sudo apt-get install libeigen3-dev
  ```

- **Boost**: Boost is a collection of C++ libraries that provide a wide range of functionality, including system operations and file system operations.

  ```bash
  sudo apt-get install libboost-dev libboost-all-dev
  ```

- **PCL (Point Cloud Library)**: PCL is an open-source library for point cloud processing. It should already be installed when you set up ROS. If not:

  ```bash
  sudo apt-get install libpcl-dev
  ```

- **Sophus**: Sophus is a C++ library for Lie groups and Lie algebra operations. You need to download and build it manually from the official website:

  - Visit the official repository: https://github.com/strasdat/Sophus

  - Follow the installation instructions provided on the repository to compile and install Sophus.

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

## Overview

![overview](https://github.com/yangsc4063/rino/blob/main/figure/overview.png)

## Primary Results

**Result on <u>KAIST02</u> of MulRan dataset:**

<div align="center">
    <img src=".\figure\result_kaist.png" alt="result_kaist" width="60%" />
</div>

**Result on <u>2021-11-28-09-18</u> of MulRan dataset:**

https://github.com/user-attachments/assets/7aafe2a6-03fb-4cba-80e3-fece956c6a8e

## Acknowledgement

Many thanks to these excellent projects:

- [ORORA](https://github.com/url-kaist/outlier-robust-radar-odometry)

- [SLAM in autonomous driving](https://github.com/gaoxiang12/slam_in_autonomous_driving)
