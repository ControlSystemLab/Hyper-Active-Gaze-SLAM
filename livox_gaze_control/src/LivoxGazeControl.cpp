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
#include <std_msgs/msg/int32.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <example_interfaces/msg/float32_multi_array.hpp>
#include <example_interfaces/msg/float32.hpp>
#include "custom_msgs/msg/i_control.hpp"
#include "custom_msgs/msg/i_status.hpp"

#include <livox_interfaces/msg/custom_msg.hpp>

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
#include <tf2_ros/buffer.h>
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
#include <chrono>

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
            double yaw_angle_carto_glb = 0.0;

            rclcpp::Subscription<livox_interfaces::msg::CustomMsg>::SharedPtr subLaserCloud;
            rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subGimbalImu;
            rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom;

            rclcpp::Publisher<example_interfaces::msg::Float32MultiArray>::SharedPtr pubRK;
            rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr pubGrid;
            rclcpp::Publisher<example_interfaces::msg::Float32>::SharedPtr pubGP;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubColorPCEdge;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubColorPCPlane;
            rclcpp::Publisher<custom_msgs::msg::IControl>::SharedPtr pubCommandOmega;
            rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr pubEdgeCount;

            std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
            std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
            


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
            geometry_msgs::msg::TransformStamped odom_carto_last;
            Eigen::Matrix<double,6,1> incremental_odom;

            std::deque<double> yaw_car_deque;
            std::deque<double> yaw_head_deque;

            double yaw_car_last = 0.0;
            double best_view_last = -10.0;
            double last_command_angle = 0.0;
            
            double global_position_command = 0.0;

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
            
            bool publish_color_cloud = true;

            bool working_mode = 1; //0 for stabilization, 1 for optimal gaze
            bool first_IMU = 1;
            bool use_carto_odom = 1;
            bool use_lio_odom = 0;

            double start_yaw_head_imu;
            double start_yaw_head_carto;
            double start_yaw_head_lio;

            double yaw_imu_last;
            double yaw_carto_last;
            double yaw_lio_last;
            bool last_plus = 1;




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
                pubColorPCEdge = create_publisher<sensor_msgs::msg::PointCloud2>("/colored_edge_point",5);
                pubColorPCPlane = create_publisher<sensor_msgs::msg::PointCloud2>("/colored_plane_point",5);
                pubCommandOmega = create_publisher<custom_msgs::msg::IControl>("/chassis_cmd", 10);
                pubEdgeCount = create_publisher<std_msgs::msg::Int32>("edge_count", 10);

                subLaserCloud = create_subscription<livox_interfaces::msg::CustomMsg>(
                        "/livox/lidar", 10,
                        std::bind(&LivoxGazeControl::livoxCallBack, this, std::placeholders::_1));

                subOdom = create_subscription<nav_msgs::msg::Odometry>(
                        "/aft_mapped_to_init", 100,
                        std::bind(&LivoxGazeControl::odometryCallBack, this, std::placeholders::_1));

                subGimbalImu = create_subscription<sensor_msgs::msg::Imu>(
                        "/gimbal_imu", rclcpp::SensorDataQoS(),
                        std::bind(&LivoxGazeControl::gimbalImuCallBack, this, std::placeholders::_1));

                tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
                tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

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
            
            
            void publish_color_pointcloud(const pcl::PointCloud<PointType>::Ptr& laserCornerCloud_, const pcl::PointCloud<PointType>::Ptr& laserPlaneCloud_){
            	pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorCloudEdge(new pcl::PointCloud<pcl::PointXYZRGB>());
            	pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorCloudPlane(new pcl::PointCloud<pcl::PointXYZRGB>());
            	
            	for(int i = 0; i < laserCornerCloud_->size(); i++){
            		pcl::PointXYZRGB point;
            		point.x = laserCornerCloud_->points[i].x;
            		point.y = laserCornerCloud_->points[i].y;
            		point.z = laserCornerCloud_->points[i].z;
            		point.r = 255;
            		point.g = 0;
            		point.b = 0;
            		colorCloudEdge->push_back(point);
            	}
            	
            	for(int i = 0; i < laserPlaneCloud_->size(); i++){
            		pcl::PointXYZRGB point;
            		point.x = laserPlaneCloud_->points[i].x;
            		point.y = laserPlaneCloud_->points[i].y;
            		point.z = laserPlaneCloud_->points[i].z;
            		point.r = 0;
            		point.g = 0;
            		point.b = 255;
            		colorCloudPlane->push_back(point);
            	}
            	
            	sensor_msgs::msg::PointCloud2 edgeCloud2;
            	sensor_msgs::msg::PointCloud2 planeCloud2;
            	
            	pcl::toROSMsg(*colorCloudEdge, edgeCloud2);
            	pcl::toROSMsg(*colorCloudPlane, planeCloud2);
            	
            	rclcpp::Time time = rclcpp::Clock().now();
            	
            	edgeCloud2.header.frame_id = "map";
            	edgeCloud2.header.stamp = time;
            	planeCloud2.header.frame_id = "map";
            	planeCloud2.header.stamp = time;
            	
            	pubColorPCEdge->publish(edgeCloud2);
            	pubColorPCPlane->publish(planeCloud2);
            	
            	
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
                robot_movement.x() = - odometry.coeff(0,0);
                robot_movement.y() = - odometry.coeff(1,0);
                map_last.move(robot_movement);

                for (grid_map::GridMapIterator it(map_last); !it.isPastEnd(); ++it) {
                    if(std::isnan(map_last.at("rob_cent_map", *it)))
                        map_last.at("rob_cent_map", *it) = 0.0;
                    else
                        map_last.at("rob_cent_map", *it) /= 1.01;
                }

                if(kalman_map_enable){
                    rotation_s_in_m << std::cos(yaw_angle), -std::sin(yaw_angle), 0.0,
                                        std::sin(yaw_angle), std::cos(yaw_angle), 0.0,
                                        0.0, 0.0, 1.0;
                    //jacob_f_rs = -rotation_s_in_m;
                    jacob_f_rs = Eigen::Matrix3f::Identity();
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
                double current_view_angle_upper = current_leading_view_angle_degree + 10.0;
                double current_view_angle_lower = current_leading_view_angle_degree - 10.0;
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
                                }else if (kalman_map_enable && map_last.at("variance_z", *iterator) > 0.0){
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

            void gimbalImuCallBack(const sensor_msgs::msg::Imu::SharedPtr msg){
                tf2::Quaternion q_gimbalhead;
                tf2::fromMsg(msg->orientation, q_gimbalhead);
                double roll_head, pitch_head, yaw_head;
                tf2::Matrix3x3(q_gimbalhead).getRPY(roll_head, pitch_head, yaw_head);

                double yaw_carto_current, yaw_lio_current;

                if(yaw_head_deque.size() > 1)
                    yaw_head_deque.pop_front();
                yaw_head_deque.push_back(yaw_head);

               if(working_mode == 0){
                    if(first_IMU){
                      start_yaw_head_imu = yaw_head;
                      yaw_imu_last = yaw_head;
                      if(use_carto_odom){
                          geometry_msgs::msg::TransformStamped transform_carto =
                          tf_buffer_->lookupTransform(
                          "map", "laser",
                          tf2::TimePointZero,tf2::durationFromSec(0.1));
                          // Convert geometry_msgs quaternion to tf2 quaternion
                          tf2::Quaternion q_;
                          tf2::fromMsg(transform_carto.transform.rotation, q_);

                          // Convert to RPY
                          tf2::Matrix3x3 m_(q_);
                          double roll_, pitch_, yaw_;
                          m_.getRPY(roll_, pitch_, yaw_);
                          start_yaw_head_carto = yaw_;
                          yaw_carto_current = yaw_;
                          yaw_carto_last = yaw_head;
                      }
                        if(use_lio_odom){
                            geometry_msgs::msg::TransformStamped transform_lio =
                            tf_buffer_->lookupTransform(
                            "camera_init", "aft_mapped",
                            tf2::TimePointZero, tf2::durationFromSec(0.1));
                            // Convert geometry_msgs quaternion to tf2 quaternion
                            tf2::Quaternion q_;
                            tf2::fromMsg(transform_lio.transform.rotation, q_);

                            // Convert to RPY
                            tf2::Matrix3x3 m_(q_);
                            double roll_, pitch_, yaw_;
                            m_.getRPY(roll_, pitch_, yaw_);
                            start_yaw_head_carto = yaw_;
                            yaw_lio_current = yaw_;
                            yaw_lio_last = yaw_;
                        }
                      first_IMU = false;
                    }
                    else{
                      double delta_yaw_head, delta_yaw_carto, delta_yaw_lio;
                      double yaw_imu_current = yaw_head;
                      delta_yaw_head = yaw_imu_current - yaw_imu_last;
                      if(delta_yaw_head > M_PI)
                        delta_yaw_head -= 2*M_PI;
                      if(delta_yaw_head < -M_PI)
                        delta_yaw_head += 2*M_PI;

                      if(use_carto_odom){
                          delta_yaw_carto = yaw_carto_current - yaw_carto_last;
                          if(delta_yaw_carto > M_PI)
                              delta_yaw_carto -= 2*M_PI;
                          if(delta_yaw_carto < -M_PI)
                              delta_yaw_carto += 2*M_PI;
                      }

                      if(use_lio_odom){
                        delta_yaw_lio = yaw_lio_current - yaw_lio_last;
                        if(delta_yaw_lio > M_PI)
                            delta_yaw_lio -= 2*M_PI;
                        if(delta_yaw_lio < -M_PI)
                            delta_yaw_lio += 2*M_PI;
                      }

                      double delta_yaw_average;
                      if(use_lio_odom && use_carto_odom){
                        delta_yaw_average = 0.4 * delta_yaw_head + 0.4 * delta_yaw_carto + 0.2 * delta_yaw_lio;
                      }else if(use_carto_odom){
                          delta_yaw_average = 0.5 * delta_yaw_head + 0.5 * delta_yaw_carto;
                      }else{
                          delta_yaw_average = delta_yaw_head;
                      }
                      double omega_head_yaw = delta_yaw_average / 1000;//imu is 1000hz

                      custom_msgs::msg::IControl command_omega;
                      command_omega.type = 0;
                      command_omega.cmd = - omega_head_yaw;
                      pubCommandOmega->publish(command_omega);
                    }
              }
//               else{
//                   float best_view_angle_rad;
//                   best_view_angle_rad = (best_view_angle_global) * (M_PI / 180.0);
//
//                   if(best_view_angle_rad > M_PI)
//                   	best_view_angle_rad -= 2 * M_PI;
//                   if(best_view_angle_rad < -M_PI)
//                   	best_view_angle_rad += 2 * M_PI;
//
//                   if(yaw_angle > M_PI)
//                   	yaw_angle -= 2 * M_PI;
//                   if(yaw_angle < -M_PI)
//                   	yaw_angle += 2 * M_PI;
//
//
//                   custom_msgs::msg::IControl command_omega;
//                   command_omega.type = 1;
//                   if((best_view_angle_rad - yaw_angle) > 0.05){
//                   	global_position_command += 0.0005;
//                   	command_omega.cmd = global_position_command;
//                   	pubCommandOmega->publish(command_omega);
//                   }else if((best_view_angle_rad - yaw_angle) < -0.05){
//                   	global_position_command -= 0.0005;
//                   	command_omega.cmd = global_position_command;
//                   	pubCommandOmega->publish(command_omega);
//                   }
//                   //pubCommandOmega->publish(command_omega);
//               }

            }

            void odometryCallBack(const nav_msgs::msg::Odometry::SharedPtr msg){
            	if(!use_carto_odom){
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
                }else{
                	if(!last_odometry){
		            last_odometry = msg;
		            odom_carto_last =tf_buffer_->lookupTransform(
                          		"map", "laser",
                          		tf2::TimePointZero,tf2::durationFromSec(0.1));
		            incremental_odom << 0,0,0,0,0,0;
		            return;
		        }
		        double delta_x = msg->pose.pose.position.x - last_odometry->pose.pose.position.x;
		        double delta_y = msg->pose.pose.position.y - last_odometry->pose.pose.position.y;
		        double delta_z = msg->pose.pose.position.z - last_odometry->pose.pose.position.z;
		        
		        geometry_msgs::msg::TransformStamped odom_carto =
                          tf_buffer_->lookupTransform(
                          "map", "laser",
                          tf2::TimePointZero,tf2::durationFromSec(0.1));
                          
                        double delta_x_carto = odom_carto.transform.translation.x - odom_carto_last.transform.translation.x;
		        double delta_y_carto = odom_carto.transform.translation.y - odom_carto_last.transform.translation.y;
		        double delta_z_carto = odom_carto.transform.translation.z - odom_carto_last.transform.translation.z;

		        double delta_roll, delta_pitch, delta_yaw;
		        tf2::Quaternion q1, q2;

		        tf2::fromMsg(last_odometry->pose.pose.orientation, q1);
		        tf2::fromMsg(msg->pose.pose.orientation, q2);
		        
		        double delta_roll_carto, delta_pitch_carto, delta_yaw_carto;
		        tf2::Quaternion q1_carto, q2_carto;

		        tf2::fromMsg(odom_carto_last.transform.rotation, q1_carto);
		        tf2::fromMsg(odom_carto.transform.rotation, q2_carto);

		        //geometry_msgs::msg::Quaternion orientation_1 = msg->pose.pose.orientation;
		        //yaw_angle = tf2::getYaw(orientation_1);//in degree
		        double roll, pitch;
		        tf2::Matrix3x3(q2).getRPY(roll, pitch, yaw_angle);
		        
		        tf2::Matrix3x3(q2_carto).getRPY(roll, pitch, yaw_angle_carto_glb);
		        
		        //yaw_angle = 0.5*yaw_angle + 0.5*yaw_angle_carto_glb;
		        yaw_angle = yaw_angle_carto_glb;

		        tf2::Matrix3x3(q1_carto.inverse() * q2_carto).getRPY(delta_roll_carto, delta_pitch_carto, delta_yaw_carto);//in radians
		        //incremental_odom << 0.5*delta_x + 0.5*delta_x_carto, 0.5*delta_y + 0.5*delta_y_carto, 0.5*delta_z + 0.5*delta_z_carto, 0.5*delta_roll + 0.5*delta_roll_carto, 0.5*delta_pitch+0.5*delta_pitch_carto, 0.5*delta_yaw+0.5*delta_yaw_carto;
		        incremental_odom << delta_x_carto, delta_y_carto, delta_z_carto, delta_roll_carto, delta_pitch_carto, delta_yaw_carto;
		        last_odometry = msg;
		        odom_carto_last = odom_carto;
                }

                

            }

            void livoxCallBack(const livox_interfaces::msg::CustomMsg &msg) {



                laserCloud.reset(new pcl::PointCloud<PointType>);
                laserCornerCloud.reset(new pcl::PointCloud<PointType>);
                laserSurfCloud.reset(new pcl::PointCloud<PointType>);
                //laserNonFeatureCloud.reset(new pcl::PointCloud<PointType>);

                LidarFeatureExtractor lidarFeatureExtractor(6,2,100,3,150,
                                                            0.02,1,1.0,0.2);
                RipleyKCalculator ripleyKCalculator (10, 0.05);
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Begin feature extraction....");

                lidarFeatureExtractor.FeatureExtract(msg, laserCloud, laserCornerCloud, laserSurfCloud,N_SCANS,Lidar_Type);
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Retrieving information from point cloud....");
                if(publish_color_cloud)
                	publish_color_pointcloud(laserCornerCloud, laserSurfCloud);
                //RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Total corner points in current scan %ld", laserCornerCloud->size());
                //RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Total surf points in current scan %ld", laserSurfCloud->size());
                ripleyKCalculator.CalculateRipleyK(laserCornerCloud, ripleyKValueCorner, numberOfFeatureCorner, centroidPointsCorner);
                //RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Total %d K value in query", static_cast<int>(ripleyKValueCorner.size()));

                ripleyKCalculator.CalculateRipleyK(laserSurfCloud, ripleyKValuePlane, numberOfFeaturePlane, centroidPointsPlane);

                std::vector<float> normalized_ripley;
                std::vector<Eigen::Matrix<float,4,1>> mean_points;
                for(int j=0; j<ripleyKValuePlane.size(); ++j){
                    //float ripley_norm = (numberOfFeatureCorner[j]/ripleyKValueCorner[j]) + (numberOfFeaturePlane[j]/(10 * ripleyKValuePlane[j]));
                    float ripley_norm = (1000 * numberOfFeatureCorner[j]) + (numberOfFeaturePlane[j]/ 1000);//disable ripley k
                    ripley_norm /= 10000.0;
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

                int edge_count_min = 30;
                int edge_count_current = laserCornerCloud->size();

                auto edge_cnt_msg = std_msgs::msg::Int32();
                edge_cnt_msg.data = edge_count_current;
                pubEdgeCount->publish(edge_cnt_msg);
                
                if(working_mode == 1){
                    float best_view_angle_rad;
                    best_view_angle_rad = (best_view_angle_global) * (M_PI / 180.0);

                    if(best_view_angle_rad > M_PI)
                        best_view_angle_rad -= 2 * M_PI;
                    if(best_view_angle_rad < -M_PI)
                        best_view_angle_rad += 2 * M_PI;

                    if(yaw_angle > M_PI)
                        yaw_angle -= 2 * M_PI;
                    if(yaw_angle < -M_PI)
                        yaw_angle += 2 * M_PI;


                    custom_msgs::msg::IControl command_omega;
                    command_omega.type = 1;
                    
                    if((edge_count_current < edge_count_min) && last_plus){
                          global_position_command += 0.05;
                          last_plus = 1;
                          command_omega.cmd = global_position_command;
                      	  pubCommandOmega->publish(command_omega);
                      	  return;
                    }
                    
//                    if((edge_count_current < edge_count_min) && !last_plus){
//                          global_position_command += 0.05;
//                          last_plus = 0;
//                          command_omega.cmd = global_position_command;
//                      	  pubCommandOmega->publish(command_omega);
//                      	  return;
//                    }
                    
                    if((best_view_angle_rad - yaw_angle) > 0.025){
                      global_position_command += 0.01;
                      command_omega.cmd = global_position_command;
                      pubCommandOmega->publish(command_omega);
                      last_plus = 1;
                    }else if((best_view_angle_rad - yaw_angle) < -0.025){
                        global_position_command -= 0.01;
                        command_omega.cmd = global_position_command;
                        pubCommandOmega->publish(command_omega);
                        last_plus = 0;
                    }
                    //global_position_command = 0.0;
                    //pubCommandOmega->publish(command_omega);
                }

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
