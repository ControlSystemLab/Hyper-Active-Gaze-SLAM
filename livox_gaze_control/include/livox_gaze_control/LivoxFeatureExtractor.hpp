#ifndef LIVOX_LIVOXFEATUREEXTRACTOR_H
#define LIVOX_LIVOXFEATUREEXTRACTOR_H
#include "rclcpp/rclcpp.hpp"
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <livox_interfaces/msg/custom_msg.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/common.h>
#include <future>
#include "opencv2/core.hpp"
#include <pcl/features/normal_3d.h>
#include <Eigen/Dense>
#include <cmath>

#define pcl_isfinite(x) std::isfinite(x)

class LidarFeatureExtractor{
    typedef pcl::PointXYZINormal PointType;
public:
    /** \brief constructor of LidarFeatureExtractor
      * \param[in] n_scans: lines used to extract lidar features
      */
    LidarFeatureExtractor(int n_scans,int NumCurvSize,float DistanceFaraway,int NumFlat,int PartNum,float FlatThreshold,
                          float BreakCornerDis,float LidarNearestDis,float KdTreeCornerOutlierDis);

    /** \brief transform float to int
      */
    static uint32_t _float_as_int(float f){
        union{uint32_t i; float f;} conv{};
        conv.f = f;
        return conv.i;
    }

    /** \brief transform int to float
      */
    static float _int_as_float(uint32_t i){
        union{float f; uint32_t i;} conv{};
        conv.i = i;
        return conv.f;
    }
    /** \brief Filter out ground plane points
     * \param[in] cloud: input cloud
     * \param[in] filtered_cloud:cloud without ground
     */
    bool filterGroundRANSAC(
        const pcl::PointCloud<PointType>::Ptr& cloud,
        pcl::PointCloud<PointType>::Ptr& filtered_cloud);

    /** \brief Determine whether the point_list is flat
      * \param[in] point_list: points need to be judged
      * \param[in] plane_threshold
      */
    bool plane_judge(const std::vector<PointType>& point_list,const int plane_threshold);

    /** \brief Detect lidar feature points
      * \param[in] cloud: original lidar cloud need to be detected
      * \param[in] pointsLessSharp: less sharp index of cloud
      * \param[in] pointsLessFlat: less flat index of cloud
      */
    void detectFeaturePoint(pcl::PointCloud<PointType>::Ptr& cloud,
                            std::vector<int>& pointsLessSharp,
                            std::vector<int>& pointsLessFlat);



    /** \brief Detect lidar feature points of CustomMsg
      * \param[in] msg: original CustomMsg need to be detected
      * \param[in] laserCloud: transform CustomMsg to pcl point cloud format
      * \param[in] laserConerFeature: less Coner features extracted from laserCloud
      * \param[in] laserSurfFeature: less Surf features extracted from laserCloud
      */
    void FeatureExtract(const livox_interfaces::msg::CustomMsg &msg,
    			pcl::PointCloud<PointType>::Ptr& laserCloud,
                        pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                        pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                        int Used_Line = 1,const int lidar_type=0);


private:
    /** \brief lines used to extract lidar features */
    const int N_SCANS;

    /** \brief store original points of each line */
    std::vector<pcl::PointCloud<PointType>::Ptr> vlines;

    /** \brief store corner feature index of each line */
    std::vector<std::vector<int>> vcorner;

    /** \brief store surf feature index of each line */
    std::vector<std::vector<int>> vsurf;

    int thNumCurvSize;

    float thDistanceFaraway;

    int thNumFlat;

    int thPartNum;

    float thFlatThreshold;

    float thBreakCornerDis;

    float thLidarNearestDis;
};

#endif //LIVOX_LIVOXFEATUREEXTRACTOR_H
