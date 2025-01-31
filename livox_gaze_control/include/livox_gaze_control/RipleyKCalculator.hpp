//
// Created by luke on 3/5/24.
//

#ifndef LIVOX_GAZE_CONTROL_RIPLEYKCALAULATOR_H
#define LIVOX_GAZE_CONTROL_RIPLEYKCALAULATOR_H

#include <cmath>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <Eigen/Dense>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

class RipleyKCalculator{
    typedef pcl::PointXYZINormal PointType;
public:
    /** \brief constructor of LidarFeatureExtractor
      * \param[in] n_scans: lines used to extract lidar features
      */
    RipleyKCalculator(int n_bins, float radius);

    /** \brief transform float to int
      */

    /** \brief Detect lidar feature points of CustomMsg
      * \param[in] msg: original CustomMsg need to be detected
      * \param[in] laserCloud: transform CustomMsg to pcl point cloud format
      * \param[in] laserConerFeature: less Coner features extracted from laserCloud
      * \param[in] laserSurfFeature: less Surf features extracted from laserCloud
      */
    void CalculateRipleyK(const pcl::PointCloud<PointType>::Ptr& laserCloud,
                          std::vector<float>& ripleyKValue,
                          std::vector<int>& numberOfFeature,
                          std::vector<Eigen::Matrix<float,4,1>>& centroidPoints);


private:
    /** \brief lines used to extract lidar features */
    const int N_BINS;

    /** \brief store original points of each line */
    std::vector<pcl::PointCloud<PointType>::Ptr> vbins;

    float Radius;
};
#endif //LIVOX_GAZE_CONTROL_RIPLEYKCALAULATOR_H
