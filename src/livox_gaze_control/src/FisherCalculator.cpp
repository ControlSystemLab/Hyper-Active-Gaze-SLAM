//
// Created by luke on 3/21/24.
//
#include "livox_gaze_control/FisherCalculator.hpp"

RipleyKCalculator::RipleyKCalculator(int n_bins, float radius)
        :N_BINS(n_bins){
    vbins.resize(N_BINS);
}

void RipleyKCalculator::CalculateRipleyK(const pcl::PointCloud<PointType>::Ptr& laserCloud,
                                         std::vector<float>& fisherValue){
    std::size_t cloud_num = laserCloud->size();
    for(auto &ptr:vbins){
        ptr.reset(new pcl::PointCloud<PointType>());
    }
    for(std::size_t i=0; i<cloud_num; ++i){
        float theta = std::atan(laserCloud->points[i].y/laserCloud->points[i].x);
        int bin_idx = int((theta*180/3.1415) - (-35.2))%10;
        vbins[bin_idx]->push_back(laserCloud->points[i]);
    }

    for(int iter=0;iter<vbins.size();++iter){
        int clouds_in_bin = vbins[iter]->size();
        numberOfFeature.push_back(clouds_in_bin);

        pcl::KdTreeFLANN<PointType> kdtree;
        kdtree.setInputCloud(vbins[iter]);

        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        float k_value = 0;
        for(int j =0; j<clouds_in_bin; ++j){
            auto searchPoint = vbins[iter]->points[j];
            if(kdtree.radiusSearch(searchPoint, Radius, pointIdxRadiusSearch, pointRadiusSquaredDistance))
                k_value += pointIdxRadiusSearch.size();
        }

        PointType min_p, max_p;
        pcl::getMinMax3D (*vbins[iter], min_p, max_p);
        float box_x = max_p.x - min_p.x;
        float box_y = max_p.y - min_p.y;
        float box_z = max_p.z - min_p.z;
        float box_volume = box_x * box_y * box_z;

        k_value = std::abs(box_volume)*(k_value/(clouds_in_bin*clouds_in_bin));
        k_value = std::abs(k_value - (4*3.1415*Radius*Radius*Radius)/3);
        ripleyKValue.push_back(k_value);

        Eigen::Matrix<float,4,1> centroid_pt;
        pcl::compute3DCentroid(*vbins[iter], centroid_pt);
        centroidPoints.push_back(centroid_pt);
        vbins[iter].reset(new pcl::PointCloud<PointType>);
    }


