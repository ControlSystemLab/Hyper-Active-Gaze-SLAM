//
// Created by luke on 3/5/24.
//
#include "livox_gaze_control/LivoxFeatureExtractor.hpp"
#include "livox_gaze_control/RipleyKCalculator.hpp"
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <example_interfaces/msg/float32_multi_array.hpp>
#include <example_interfaces/msg/float32.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <math.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#include <Eigen/Dense>

typedef pcl::PointXYZINormal PointType;
class LivoxGazeControl : public rclcpp::Node
        {
        public:
            int Lidar_Type = 0;
            int N_SCANS = 1;
            int NumCurvSize = 2;
            float DistanceFaraway = 100;
            int NumFlat = 3;
            int PartNum = 150;
            float FlatThreshold = 0.02; // cloud curvature threshold of flat feature
            float BreakCornerDis = 1; // break distance of break points
            float LidarNearestDis = 1.0; // if(depth < LidarNearestDis) do not use this point
            float KdTreeCornerOutlierDis = 0.2; // corner filter threshold

            int N_BINS = 10;
            float Radius = 0.05;

            double yaw_angle = 0.0;

            rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr subLaserCloud;
            rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subBodyImu;
            rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom;

            rclcpp::Publisher<example_interfaces::msg::Float32MultiArray>::SharedPtr pubRK;
            rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr pubGrid;
            rclcpp::Publisher<example_interfaces::msg::Float32>::SharedPtr pubGP;


            pcl::PointCloud<PointType>::Ptr laserCloud;
            pcl::PointCloud<PointType>::Ptr laserCornerCloud;
            pcl::PointCloud<PointType>::Ptr laserSurfCloud;
            pcl::PointCloud<PointType>::Ptr laserNonFeatureCloud;

            std::vector<float> ripleyKValueCorner;

            std::vector<int> numberOfFeatureCorner;

            std::vector<Eigen::Matrix<float,4,1>> centroidPointsCorner;

            std::vector<float> ripleyKValuePlane;

            std::vector<int> numberOfFeaturePlane;

            std::vector<Eigen::Matrix<float,4,1>> centroidPointsPlane;

            grid_map::GridMap map_last;

            grid_map::GridMap map_curr;

            bool map_init = false;

            nav_msgs::msg::Odometry::SharedPtr last_odometry;
            Eigen::Matrix<double,6,1> incremental_odom;

            std::deque<double> yaw_car_deque;

            double yaw_car_last = 0.0;
            double best_view_last = -10.0;
            double last_command_angle = 0.0;

            double best_view_angle_global;

            bool kalman_map_enable = true;
            bool kalman_map_init = false;
            //matrix for map frame update
            Eigen::Matrix3f cov_last;
            Eigen::Matrix3f cov_current;
            Eigen::Matrix3f cov_odom;
            Eigen::Matrix3f jacob_f_rp;
            Eigen::Matrix3f jacob_f_rs;
            //matrix for measurement update
            Eigen::Matrix3f cov_r_lower;
            Eigen::Matrix3f cov_m;
            Eigen::Matrix3f cov_r_upper;
            Eigen::Matrix3f jacob_g_rlower;
            Eigen::Matrix3f jacob_g_rupper;

            Eigen::Matrix3f rotation_s_in_m;
            Eigen::Vector3f variance_odom_trans;
            Eigen::Vector3f variance_odom_rot;
            Eigen::Vector3f variance_meas;

            float height_last;
            float height_now;
            float height_variance_last;
            float height_variance_now;




            LivoxGazeControl(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("livox_gaze_control", options)
            {
                ripleyKValueCorner.reserve(10);
                numberOfFeatureCorner.reserve(10);
                centroidPointsCorner.reserve(10);
                ripleyKValuePlane.reserve(10);
                numberOfFeaturePlane.reserve(10);
                centroidPointsPlane.reserve(10);

                /*Eigen::Matrix<float,4,1> init_point;
                init_point << 0,0,0,0;
                ripleyKValueCorner(10,0);
                numberOfFeatureCorner(10,0);
                centroidPointsCorner(10,init_point);
                ripleyKValuePlane(10,0);
                numberOfFeaturePlane(10,0);
                centroidPointsPlane(10,init_point);*/

                pubRK = create_publisher<example_interfaces::msg::Float32MultiArray>("/RK", 10);
                pubGrid = create_publisher<grid_map_msgs::msg::GridMap>("/grid_map", 1);
                pubGP = create_publisher<example_interfaces::msg::Float32>("/gimbal_pos", 1);

                subLaserCloud = create_subscription<livox_ros_driver2::msg::CustomMsg>(
                        "/livox/lidar", 0,
                        std::bind(&LivoxGazeControl::livoxCallBack, this, std::placeholders::_1));

                subOdom = create_subscription<nav_msgs::msg::Odometry>(
                        "/Odometry", 0,
                        std::bind(&LivoxGazeControl::odometryCallBack, this, std::placeholders::_1));

                subBodyImu = create_subscription<sensor_msgs::msg::Imu>(
                        "/bodyimu/data", 0,
                        std::bind(&LivoxGazeControl::bodyImuCallBack, this, std::placeholders::_1));

            }
            std::pair<int, int> find_end_index(std::pair<int,int> start_index, double length, double bearing_angle_in_deg,
                                          double grid_resolution){
                double angle_in_rad = bearing_angle_in_deg * M_PI / 180.0;

                double horizontal_displacement = length * cos(angle_in_rad) / grid_resolution;
                double vertical_displacement = length * sin(angle_in_rad) / grid_resolution;

                double end_x = round(start_index.first + horizontal_displacement);
                double end_y = round(start_index.second + vertical_displacement);

                return std::make_pair(end_x, end_y);
            }

            void build_grid_map(const std::vector<float> &ripleyKValue,
                                const std::vector<Eigen::Matrix<float,4,1>> &centroidPoints,
                                const Eigen::Matrix<double,6,1> &odometry){

                if(!map_init){
                    map_last.add("rob_cent_map");
                    if(kalman_map_enable) {
                        map_last.add("variance_x");
                        map_last.add("variance_y");
                        map_last.add("variance_z");
                    }
                    map_last.setFrameId("map");
                    map_last.setGeometry(grid_map::Length(30.0, 30.0), 0.3);

                    if(kalman_map_enable) {
                        variance_meas << 1e-1, 1e-1, 10;
                        variance_odom_rot << 1e-1, 1e-1, 3e-1;
                        variance_odom_trans << 3e-1, 3e-1, 1e-1;
                        cov_odom = variance_odom_trans.asDiagonal();
                        cov_m = variance_meas.asDiagonal();
                        cov_r_upper = variance_odom_rot.asDiagonal();
                        cov_last = cov_odom;
                        cov_current = cov_last;
                    }


                    for (grid_map::GridMapIterator it(map_last); !it.isPastEnd(); ++it) {
                        map_last.at("rob_cent_map", *it) = 0.0;
                        if(kalman_map_enable) {
                            map_last.at("variance_x", *it) = cov_last(0, 0);
                            map_last.at("variance_y", *it) = cov_last(1, 1);
                            map_last.at("variance_z", *it) = cov_last(2, 2);
                        }
                    }

                    map_init = true;
                    for (int i = 0; i < centroidPoints.size(); ++i){
                        grid_map::Position position;
                        position.x() = centroidPoints[i].coeff(0,0);
                        position.y() = centroidPoints[i].coeff(1,0);
                        if((std::sqrt(position.x()*position.x() + position.y()*position.y())) < 10.0){
                            if(std::isnan(position.x()) || std::isnan(position.y()) || std::isnan(ripleyKValue[i]))
                                continue;
                            map_last.atPosition("rob_cent_map", position) = ripleyKValue[i];
                            if(kalman_map_enable){
                                Eigen::Vector3f r_s(position.x(),position.y(),1.0);
                                Eigen::Matrix3f r_s_skew;
                                r_s_skew << 0, -r_s(2),r_s(1),
                                            r_s(2), 0, -r_s(0),
                                            -r_s(1), r_s(0), 0;
                                rotation_s_in_m = Eigen::Matrix3f::Identity();
                                jacob_g_rlower = rotation_s_in_m;
                                jacob_g_rupper = - rotation_s_in_m * r_s_skew;
                                cov_r_lower = jacob_g_rlower * cov_m * jacob_g_rlower.transpose() + jacob_g_rupper * cov_r_upper * jacob_g_rupper.transpose();
                                map_last.atPosition("variance_x", position) = cov_r_lower(0, 0);
                                map_last.atPosition("variance_y", position) = cov_r_lower(1, 1);
                                map_last.atPosition("variance_z", position) = cov_r_lower(2, 2);
                            }
                        }
                    }
                    return;
                }

                grid_map::Position robot_movement;
                robot_movement.x() = odometry.coeff(0,0);
                robot_movement.y() = odometry.coeff(1,0);
                map_last.move(robot_movement);

                for (grid_map::GridMapIterator it(map_last); !it.isPastEnd(); ++it) {
                    if(std::isnan(map_last.at("rob_cent_map", *it)))
                        map_last.at("rob_cent_map", *it) = 0.0;
                    else
                        map_last.at("rob_cent_map", *it) /= 1.1;
                }

                if(kalman_map_enable){
                    rotation_s_in_m << std::cos(yaw_angle), -std::sin(yaw_angle), 0.0,
                                        std::sin(yaw_angle), std::cos(yaw_angle), 0.0,
                                        0.0, 0.0, 1.0;
                    jacob_f_rs = -rotation_s_in_m;
                    for (grid_map::GridMapIterator it(map_last); !it.isPastEnd(); ++it){
                        if(std::isnan(map_last.at("variance_x", *it)) || std::isnan(map_last.at("variance_y", *it)) || std::isnan(map_last.at("variance_z", *it))){
                            cov_last = variance_odom_trans.asDiagonal();
                            map_last.at("variance_x", *it) = cov_last(0, 0);
                            map_last.at("variance_y", *it) = cov_last(1, 1);
                            map_last.at("variance_z", *it) = cov_last(2, 2);

                        }else{
                            Eigen::Vector3f variance_last_vector (map_last.at("variance_x", *it), map_last.at("variance_y", *it), map_last.at("variance_z", *it));
                            cov_last = variance_last_vector.asDiagonal();
                            cov_current = cov_last + jacob_f_rs * cov_odom * jacob_f_rs.transpose();
                            map_last.at("variance_x", *it) = cov_current(0, 0);
                            map_last.at("variance_y", *it) = cov_current(1, 1);
                            map_last.at("variance_z", *it) = cov_current(2, 2);
                        }
                    }
                }

                for (int i = 0; i < centroidPoints.size(); ++i){
                    grid_map::Position position_s, position;
                    position_s.x() = centroidPoints[i].coeff(0,0);
                    position_s.y() = centroidPoints[i].coeff(1,0);

                    if((std::sqrt(position_s.x()*position_s.x() + position_s.y()*position_s.y())) < 10.0){
                        if(std::isnan(position_s.x()) || std::isnan(position_s.y()) || std::isnan(ripleyKValue[i]))
                            continue;
                        //double yaw_radians = yaw_angle * (M_PI / 180.0);
                        position.x() = std::cos(yaw_angle) * position_s.x() - std::sin(yaw_angle) * position_s.y();
                        position.y() = std::cos(yaw_angle) * position_s.y() + std::sin(yaw_angle) * position_s.x();
                        if(!kalman_map_enable) {
                            map_last.atPosition("rob_cent_map", position) = ripleyKValue[i];
                        }else{
                            height_last = map_last.atPosition("rob_cent_map", position);
                            height_variance_last = map_last.atPosition("variance_z", position);
                            Eigen::Vector3f r_s(position_s.x(),position_s.y(),1.0);
                            Eigen::Matrix3f r_s_skew;
                            r_s_skew << 0, -r_s(2),r_s(1),
                                    r_s(2), 0, -r_s(0),
                                    -r_s(1), r_s(0), 0;

                            jacob_g_rlower = rotation_s_in_m;
                            jacob_g_rupper = - rotation_s_in_m * r_s_skew;
                            cov_r_lower = jacob_g_rlower * cov_m * jacob_g_rlower.transpose() + jacob_g_rupper * cov_r_upper * jacob_g_rupper.transpose();
                            map_last.atPosition("variance_x", position) = cov_r_lower(0, 0);
                            map_last.atPosition("variance_y", position) = cov_r_lower(1, 1);
                            map_last.atPosition("variance_z", position) = cov_r_lower(2, 2);
                            height_now = (height_variance_last*ripleyKValue[i] + cov_r_lower(2,2)*height_last)/(height_variance_last + cov_r_lower(2,2));
                            height_variance_now = (height_variance_last * cov_r_lower(2,2))/(height_variance_last + cov_r_lower(2,2));
                            map_last.atPosition("variance_z", position) = height_variance_now;
                            map_last.atPosition("rob_cent_map", position) = height_now;
                        }
                    }
                }

                rclcpp::Time time = rclcpp::Clock().now();
                map_last.setTimestamp(time.nanoseconds());
                std::unique_ptr<grid_map_msgs::msg::GridMap> message;
                message = grid_map::GridMapRosConverter::toMessage(map_last);
                pubGrid->publish(std::move(message));
            }

            void optimize_gaze_angle() {
                double best_rating_in_fov, current_rating_in_fov;
                best_rating_in_fov = 0.0;
                double best_view_angle;
                grid_map::Position predict_pos;
                //predict_pos.x() = incremental_odom.coeff(0, 0);
                //predict_pos.y() = incremental_odom.coeff(1, 0);
                predict_pos.x() = 0.0;
                predict_pos.y() = 0.0;
                grid_map::Index start_index;
                map_last.getIndex(predict_pos, start_index);
                std::pair<int, int> start_index_pair = std::make_pair(start_index(0), start_index(1));

                const double pi = 3.1415926;
                double current_leading_view_angle_degree = yaw_angle * (180.0 / pi);
                double current_view_angle_upper = current_leading_view_angle_degree + 2.0;
                double current_view_angle_lower = current_leading_view_angle_degree - 2.0;
                std::vector<double> leading_view_angle_candidates;
                double interval = (current_view_angle_upper - current_view_angle_lower) / 4;
                for (int i = 0; i < 5; ++i) {
                    leading_view_angle_candidates.push_back(current_view_angle_lower + i * interval);
                    //RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "%d th view angle is %f", i, leading_view_angle_candidates[i]);
                }

                for (int j = 0; j < 5; ++j) {
                    current_rating_in_fov = 0.0;
                    double this_view_angle = leading_view_angle_candidates[j];
                    double this_view_angle_upper = this_view_angle + 35.0;
                    double this_view_angle_lower = this_view_angle - 35.0;
                    std::vector<double> this_view_angle_rays;
                    double interval_rays = (this_view_angle_upper - this_view_angle_lower) / 9;
                    for (int k = 0; k < 10; ++k) {
                        this_view_angle_rays.push_back(this_view_angle_lower + k * interval_rays);
                    }
                    for (int m = 0; m < 10; ++m) {
                        grid_map::Position end_pos_pair;
                        //end_pos_pair.x() = predict_pos.x() + 10 * std::cos(this_view_angle_rays[m] * (M_PI / 180.0));
                        //end_pos_pair.y() = predict_pos.y() + 10 * std::sin(this_view_angle_rays[m] * (M_PI / 180.0));
                        end_pos_pair.x() = 10 * std::cos(this_view_angle_rays[m] * (M_PI / 180.0));
                        end_pos_pair.y() = 10 * std::sin(this_view_angle_rays[m] * (M_PI / 180.0));
                        grid_map::Index end_index;
                        map_last.getIndex(end_pos_pair, end_index);
                        //RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "start index is (%d,%d), end index is (%d,%d)", start_index(0),start_index(1),end_index(0),end_index(1));
                        for (grid_map::LineIterator iterator(map_last, start_index, end_index); !iterator.isPastEnd();
                             ++iterator) {
                            if (map_last.at("rob_cent_map", *iterator) > 0.0) {
                                if(kalman_map_enable) {
                                    current_rating_in_fov += map_last.at("rob_cent_map", *iterator);
                                }else if (map_last.at("variance_z", *iterator) > 0.0){
                                    current_rating_in_fov += map_last.at("rob_cent_map", *iterator)/map_last.at("variance_z", *iterator);
                                }else{
                                    current_rating_in_fov += map_last.at("rob_cent_map", *iterator);
                                }
                                break;
                            }
                        }
                    }
                    //RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Current rating is %f", current_rating_in_fov);
                    if (current_rating_in_fov > best_rating_in_fov) {
                        best_rating_in_fov = current_rating_in_fov;
                        best_view_angle = this_view_angle;
                    }
                }

                best_view_angle_global = best_view_angle;
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Best rating is %f", best_rating_in_fov);
                /*float best_view_angle_rad;
                *//*if ((best_view_angle * current_leading_view_angle_degree) > 0){
                    best_view_angle_rad = (best_view_angle - current_leading_view_angle_degree) * (M_PI / 180.0);
                }else if(current_leading_view_angle_degree < 0){
                    best_view_angle_rad = (best_view_angle-180-180-current_leading_view_angle_degree) * (M_PI/180.0);
                }else{
                    best_view_angle_rad = (best_view_angle + 180 + 180 - current_leading_view_angle_degree) * (M_PI / 180.0);
                }*//*
                example_interfaces::msg::Float32 gp_msg;
                double yaw_car_current = 0;
                yaw_car_current = yaw_car_deque.back();
                double delta_yaw_car;
                if((yaw_car_current*yaw_car_last)>=0){
                    delta_yaw_car = yaw_car_current - yaw_car_last;
                }else{
                    if(yaw_car_current>0){
                        delta_yaw_car = yaw_car_current - yaw_car_last - 2*M_PI;
                    }else{
                        delta_yaw_car = yaw_car_current - yaw_car_last + 2*M_PI;
                    }
                }
                double yaw_car_preidict = yaw_car_current + delta_yaw_car;
                if(yaw_car_preidict < -M_PI)
                    yaw_car_preidict = 2*M_PI + yaw_car_preidict;
                if(yaw_car_preidict > M_PI)
                    yaw_car_preidict = yaw_car_preidict - 2*M_PI;
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Predict car yaw angle is %f", yaw_car_preidict);
                best_view_angle_rad = (best_view_angle) * (M_PI / 180.0);
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Current best view angle is %f", best_view_angle_rad);
                if((yaw_car_preidict*best_view_angle_rad)>0){
                    gp_msg.data = best_view_angle_rad - yaw_car_preidict;
                }else{
                    if(best_view_angle_rad>0){
                        gp_msg.data = best_view_angle_rad - yaw_car_preidict - 2*M_PI;
                    }else{
                        gp_msg.data = best_view_angle_rad - yaw_car_preidict + 2*M_PI;
                    }
                }
                *//*if(yaw_car_last < -3.5){
                    gp_msg.data = best_view_angle_rad;
                    yaw_car_last = yaw_car_deque.back();
                    last_command_angle = gp_msg.data;
                }
                else{
                    if((yaw_car_deque.back() * yaw_car_last) > 0){
                        delta_yaw_car = yaw_car_deque.back() - yaw_car_last;
                    }else if(yaw_car_last > 0){
                        delta_yaw_car = yaw_car_deque.back() + 3.14 + 3.14 - yaw_car_last;
                    }else{
                        delta_yaw_car = yaw_car_deque.back() - 3.14 - 3.14 - yaw_car_last;
                    }
                    yaw_car_last = yaw_car_deque.back();
                    gp_msg.data = last_command_angle - best_view_angle_rad + delta_yaw_car;
                    last_command_angle = gp_msg.data;
                }*//*
                yaw_car_last = yaw_car_current;
                pubGP->publish(gp_msg);*/
            }

            void bodyImuCallBack(const sensor_msgs::msg::Imu::SharedPtr msg){
                tf2::Quaternion q_body;
                tf2::fromMsg(msg->orientation, q_body);
                double roll_car, pitch_car, yaw_car;
                tf2::Matrix3x3(q_body).getRPY(roll_car, pitch_car, yaw_car);

                if(yaw_car_deque.size() > 1)
                    yaw_car_deque.pop_front();
                yaw_car_deque.push_back(yaw_car);

                float best_view_angle_rad;
                /*if ((best_view_angle * current_leading_view_angle_degree) > 0){
                    best_view_angle_rad = (best_view_angle - current_leading_view_angle_degree) * (M_PI / 180.0);
                }else if(current_leading_view_angle_degree < 0){
                    best_view_angle_rad = (best_view_angle-180-180-current_leading_view_angle_degree) * (M_PI/180.0);
                }else{
                    best_view_angle_rad = (best_view_angle + 180 + 180 - current_leading_view_angle_degree) * (M_PI / 180.0);
                }*/
                example_interfaces::msg::Float32 gp_msg;
                double yaw_car_current = 0;
                yaw_car_current = yaw_car_deque.back();
                double delta_yaw_car;
                if((yaw_car_current*yaw_car_last)>=0){
                    delta_yaw_car = yaw_car_current - yaw_car_last;
                }else{
                    if(yaw_car_current>0){
                        delta_yaw_car = yaw_car_current - yaw_car_last - 2*M_PI;
                    }else{
                        delta_yaw_car = yaw_car_current - yaw_car_last + 2*M_PI;
                    }
                }
                double yaw_car_preidict = yaw_car_current + delta_yaw_car;
                if(yaw_car_preidict < -M_PI)
                    yaw_car_preidict = 2*M_PI + yaw_car_preidict;
                if(yaw_car_preidict > M_PI)
                    yaw_car_preidict = yaw_car_preidict - 2*M_PI;
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Predict car yaw angle is %f", yaw_car_preidict);
                best_view_angle_rad = (best_view_angle_global) * (M_PI / 180.0);
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Current best view angle is %f", best_view_angle_rad);
                if((yaw_car_preidict*best_view_angle_rad)>0){
                    gp_msg.data = best_view_angle_rad - yaw_car_preidict;
                }else{
                    if(best_view_angle_rad>0){
                        gp_msg.data = best_view_angle_rad - yaw_car_preidict - 2*M_PI;
                    }else{
                        gp_msg.data = best_view_angle_rad - yaw_car_preidict + 2*M_PI;
                    }
                }
                /*if(yaw_car_last < -3.5){
                    gp_msg.data = best_view_angle_rad;
                    yaw_car_last = yaw_car_deque.back();
                    last_command_angle = gp_msg.data;
                }
                else{
                    if((yaw_car_deque.back() * yaw_car_last) > 0){
                        delta_yaw_car = yaw_car_deque.back() - yaw_car_last;
                    }else if(yaw_car_last > 0){
                        delta_yaw_car = yaw_car_deque.back() + 3.14 + 3.14 - yaw_car_last;
                    }else{
                        delta_yaw_car = yaw_car_deque.back() - 3.14 - 3.14 - yaw_car_last;
                    }
                    yaw_car_last = yaw_car_deque.back();
                    gp_msg.data = last_command_angle - best_view_angle_rad + delta_yaw_car;
                    last_command_angle = gp_msg.data;
                }*/
                yaw_car_last = yaw_car_current;
                pubGP->publish(gp_msg);
            }
            void odometryCallBack(const nav_msgs::msg::Odometry::SharedPtr msg){
                if(!last_odometry){
                    last_odometry = msg;
                    incremental_odom << 0,0,0,0,0,0;
                    return;
                }
                double delta_x = msg->pose.pose.position.x - last_odometry->pose.pose.position.x;
                double delta_y = msg->pose.pose.position.y - last_odometry->pose.pose.position.y;
                double delta_z = msg->pose.pose.position.z - last_odometry->pose.pose.position.z;

                double delta_roll, delta_pitch, delta_yaw;
                tf2::Quaternion q1, q2;

                tf2::fromMsg(last_odometry->pose.pose.orientation, q1);
                tf2::fromMsg(msg->pose.pose.orientation, q2);

                //geometry_msgs::msg::Quaternion orientation_1 = msg->pose.pose.orientation;
                //yaw_angle = tf2::getYaw(orientation_1);//in degree
                double roll, pitch;
                tf2::Matrix3x3(q2).getRPY(roll, pitch, yaw_angle);

                tf2::Matrix3x3(q1.inverse() * q2).getRPY(delta_roll, delta_pitch, delta_yaw);//in radians
                incremental_odom << delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw;
                last_odometry = msg;

            }

            void livoxCallBack(const livox_ros_driver2::msg::CustomMsg &msg) {



                //laserCloud.reset(new pcl::PointCloud<PointType>);
                laserCornerCloud.reset(new pcl::PointCloud<PointType>);
                laserSurfCloud.reset(new pcl::PointCloud<PointType>);
                //laserNonFeatureCloud.reset(new pcl::PointCloud<PointType>);

                LidarFeatureExtractor lidarFeatureExtractor(1,2,100,3,150,
                                                            0.02,1,1.0,0.2);
                RipleyKCalculator ripleyKCalculator (10, 0.05);

                lidarFeatureExtractor.FeatureExtract(msg, laserCornerCloud, laserSurfCloud,N_SCANS,Lidar_Type);
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Retrieving information from point cloud....");
                //RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Total corner points in current scan %ld", laserCornerCloud->size());
                //RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Total surf points in current scan %ld", laserSurfCloud->size());
                ripleyKCalculator.CalculateRipleyK(laserCornerCloud, ripleyKValueCorner, numberOfFeatureCorner, centroidPointsCorner);
                //RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Total %d K value in query", static_cast<int>(ripleyKValueCorner.size()));

                ripleyKCalculator.CalculateRipleyK(laserSurfCloud, ripleyKValuePlane, numberOfFeaturePlane, centroidPointsPlane);

                std::vector<float> normalized_ripley;
                std::vector<Eigen::Matrix<float,4,1>> mean_points;
                for(int j=0; j<ripleyKValuePlane.size(); ++j){
                    float ripley_norm = (numberOfFeatureCorner[j]/ripleyKValueCorner[j]) + (numberOfFeaturePlane[j]/(10 * ripleyKValuePlane[j]));
                    ripley_norm /= 100000.0;
                    normalized_ripley.push_back(ripley_norm);
                    Eigen::Matrix<float,4,1> mean_pt = ((numberOfFeatureCorner[j] * centroidPointsCorner[j]) + (numberOfFeaturePlane[j] * centroidPointsPlane[j]))/(numberOfFeatureCorner[j] + numberOfFeaturePlane[j]);
                    mean_points.push_back(mean_pt);
                }

                /*for(int cont=0;cont<normalized_ripley.size();cont++) {
                    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "%d K value is %f", cont, normalized_ripley[cont]);
                    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "%d centroid cordinates is %f, %f, %f", cont, mean_points[cont].coeff(0,0), mean_points[cont].coeff(1,0), mean_points[cont].coeff(2,0));
                }*/
                //RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Incremental odom is %f, %f, %f", incremental_odom.coeff(0,0),incremental_odom.coeff(1,0),incremental_odom.coeff(2,0));
                LivoxGazeControl::build_grid_map(normalized_ripley, mean_points, incremental_odom);
                LivoxGazeControl::optimize_gaze_angle();

                /*std::stringstream ss;
                ss << "Ripley K Values in Current Scan: ";
                std::copy(ripleyKValueCorner.begin(), ripleyKValueCorner.end(), std::ostream_iterator<float>(ss, " "));
                ss << std::endl;
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), ss.str().c_str());*/

                /*example_interfaces::msg::Float32MultiArray kValuesmsg;
                for(auto kValue : ripleyKValueCorner){
                    kValuesmsg.data.push_back(kValue);
                }

                pubRK->publish(kValuesmsg);*/

                //sensor_msgs::msg::PointCloud2 laserCloudMsg;
                //pcl::toROSMsg(*laserCloud, laserCloudMsg);
                //laserCloudMsg.header = msg.header;
                //laserCloudMsg.header.stamp.fromNSec(msg->timebase+msg->points.back().offset_time);
                //pubFullLaserCloud.publish(laserCloudMsg);

                /*std::fill(ripleyKValueCorner.begin(), ripleyKValueCorner.end(), 0);
                memset(&ripleyKValueCorner[0],0.0,ripleyKValueCorner.size() * sizeof ripleyKValueCorner[0]);
                std::fill(numberOfFeatureCorner.begin(), numberOfFeatureCorner.end(), 0);
                std::fill(ripleyKValuePlane.begin(), ripleyKValuePlane.end(), 0);
                std::fill(numberOfFeaturePlane.begin(), numberOfFeaturePlane.end(), 0);*/

                ripleyKValueCorner.clear();
                numberOfFeatureCorner.clear();
                centroidPointsCorner.clear();
                ripleyKValuePlane.clear();
                numberOfFeaturePlane.clear();
                centroidPointsPlane.clear();



            }
        };


int main(int argc, char** argv){
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;

    auto LC = std::make_shared<LivoxGazeControl>(options);

    exec.add_node(LC);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Gaze control start....");
    exec.spin();

    rclcpp::shutdown();
    return 0;
}
