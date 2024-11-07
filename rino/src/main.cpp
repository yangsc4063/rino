//
// Created by Yangsc on 23-10-10.
//

#include <ros/ros.h>

#include "rino.hpp"


int main(int argc, char** argv) {

    ros::init(argc, argv, "rino");

    RINO rino;

    ros::spin();

    return 0;
}