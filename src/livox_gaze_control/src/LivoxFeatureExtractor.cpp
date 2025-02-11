//
// Created by luke on 3/1/24.
//
#include "livox_gaze_control/LivoxFeatureExtractor.hpp"
typedef pcl::PointXYZINormal PointType;

LidarFeatureExtractor::LidarFeatureExtractor(int n_scans,int NumCurvSize,float DistanceFaraway,int NumFlat,
                                             int PartNum,float FlatThreshold,float BreakCornerDis,float LidarNearestDis,float KdTreeCornerOutlierDis)
        :N_SCANS(n_scans),
         thNumCurvSize(NumCurvSize),
         thDistanceFaraway(DistanceFaraway),
         thNumFlat(NumFlat),
         thPartNum(PartNum),
         thFlatThreshold(FlatThreshold),
         thBreakCornerDis(BreakCornerDis),
         thLidarNearestDis(LidarNearestDis){
    vlines.resize(N_SCANS);
    for(auto & ptr : vlines){
        ptr.reset(new pcl::PointCloud<PointType>());
    }
    vcorner.resize(1000);
    vsurf.resize(1000);
}






void LidarFeatureExtractor::FeatureExtract(const livox_ros_driver2::msg::CustomMsg &msg,
                                           pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                           pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                           const int Used_Line,const int lidar_type){

    float cloudCurvature[400000];
    int cloudSortInd[400000];
    int cloudNeighborPicked[400000]; // not picked 0, picked 1(不能作为特征点)
    // cloudLabel:
    // normal 0, curvature < kThresholdFlat -1, too far or too near 99,
    // edgePointsLessSharp 1, edgePointsSharp 2
    int cloudLabel[400000];

    int cloudSize = msg.point_num;

    PointType point;
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>);
    for (int i = 0; i < cloudSize; i++) {
        point.x = msg.points[i].x;
        point.y = msg.points[i].y;
        point.z = msg.points[i].z;
        point.intensity = msg.points[i].reflectivity; // intensity 整数是scan的数目，小数是时间戳

        laserCloud->push_back(point);
    }

    int kNearestNeighbor = 10;
    std::vector<int> kNeighborIndex(kNearestNeighbor);
    std::vector<float> kNeighborDistance(kNearestNeighbor);

    pcl::KdTreeFLANN<PointType> kdcloud;
    kdcloud.setInputCloud(laserCloud);

    // 3. 三个内容: 1. 计算曲率 2. 初始化cloudSortInd, cloudNeighborPicked, cloudLabel 3. unreliable points(太近或太远)
    for (int i = 5; i < cloudSize - 5; i++) {
        float dis = sqrt(laserCloud->points[i].x * laserCloud->points[i].x +
                         laserCloud->points[i].y * laserCloud->points[i].y +
                         laserCloud->points[i].z * laserCloud->points[i].z);



        //int kNumCurvSize = 0;
        float diffX = 0, diffY = 0, diffZ = 0;
        for (int j = 1; j <= 5; ++j) {
            //if((laserCloud->points[i - j].x - laserCloud->points[i + j].x)<0.2 && (laserCloud->points[i - j].y - laserCloud->points[i + j].y)<0.2 && (laserCloud->points[i - j].z - laserCloud->points[i + j].z)<0.2) {
            diffX += laserCloud->points[i - j].x + laserCloud->points[i + j].x;
            diffY += laserCloud->points[i - j].y + laserCloud->points[i + j].y;
            diffZ += laserCloud->points[i - j].z + laserCloud->points[i + j].z;


        }
        diffX -= 2 * 5 * laserCloud->points[i].x;
        diffY -= 2 * 5 * laserCloud->points[i].y;
        diffZ -= 2 * 5 * laserCloud->points[i].z;


        float tmp2 = diffX * diffX + diffY * diffY + diffZ * diffZ;
        // 在一个邻域内计算扫描方向点云的Laplacian算子，作为点云在该点的曲率
        cloudCurvature[i] = tmp2;
        cloudSortInd[i] = i;
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;

        /// Mark un-reliable points
        constexpr float kMaxFeatureDis = 1e4;
        if (fabs(dis) > 7.0 || fabs(dis) < 1e-4 || !std::isfinite(dis)) {
            cloudLabel[i] = 99;
            cloudNeighborPicked[i] = 1; // 标签为1不能作为特征点
        }

        if((laserCloud->points[i].z < -0.03) || (laserCloud->points[i].z > 0.3))
            cloudNeighborPicked[i] = 1; // 标签为1不能作为特征点
    }

    // 4. 一个点距前一个点和距后一个点的距离都太远就舍弃(尖锐的点，或是某种空间上的噪点)
    // 一个点距前一个点和距后一个点的距离都太远就舍弃(尖锐的点)
    for (int i = 10; i < cloudSize - 10; i++) {
        for (int j = 1; j<5; j++) {
            float diffX = laserCloud->points[i + j].x - laserCloud->points[i].x;
            float diffY = laserCloud->points[i + j].y - laserCloud->points[i].y;
            float diffZ = laserCloud->points[i + j].z - laserCloud->points[i].z;
            float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;

            float diffX2 = laserCloud->points[i].x - laserCloud->points[i - j].x;
            float diffY2 = laserCloud->points[i].y - laserCloud->points[i - j].y;
            float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - j].z;
            float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

            if (diff > 0.05 || diff2 > 0.05) {
                cloudNeighborPicked[i] = 1;
                break;
            }
        }
    }




    if (true) {
        float kThresholdFlat = 0.01;
        float kThresholdSharp = 0.1;
    }

    float t_q_sort = 0;
    // kNumCurvSize计算曲率的邻居数6； kNumRegion：把一个扫描线再分成若干区域(50)

    pcl::PointCloud<PointType>::Ptr planarPointsLessFlatScan(new pcl::PointCloud<PointType>);

    // start point
    int sp = 0;
    int ep = cloudSize;


    // sort the curvatures from small to large： 按照曲率由小到大排序，不改变点，改变id
    //bubble sorting
    for (int k = sp; k < ep; k++) {
        for (int l = k + 1; l < ep; l++) {
            if (cloudCurvature[cloudSortInd[l]] < cloudCurvature[cloudSortInd[k]])
            {
                int temp = cloudSortInd[k];
                cloudSortInd[k] = cloudSortInd[l];
                cloudSortInd[l] = temp;
            }
        }
    }

    float SumCurRegion = 0.0;
    float MaxCurRegion = cloudCurvature[cloudSortInd[ep]];  //the largest curvature in sp ~ ep

    // ////////////////////////////////////////////////////////////////////////
    // 提取edge point
    // ////////////////////////////////////////////////////////////////////////
    // 提取edge point
    // ////////////////////////////////////////////////////////////////////////
    int largestPickedNum = 0;
    for (int k = ep; k >= sp; k--) {
        int ind = cloudSortInd[k];

        // 提取edge points(曲率很大的点)
        if ((cloudCurvature[ind] > 0.5) && (cloudNeighborPicked[ind] != 1)) {

            cloudLabel[ind] = 1;

            PointType searchPoint = laserCloud->points[ind];
            kdcloud.nearestKSearch(searchPoint, kNearestNeighbor, kNeighborIndex, kNeighborDistance);
            Eigen::Matrix3f covariance_matrix;

            Eigen::Matrix<float,4,1> searchPt;
            searchPt << searchPoint.x, searchPoint.y, searchPoint.z, 1.0;
            pcl::computeCovarianceMatrixNormalized(*laserCloud, kNeighborIndex, searchPt, covariance_matrix);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
            float smallEigenValue, midEigenValue, bigEigenValue;
            smallEigenValue = eigen_solver.eigenvalues().coeff(0); midEigenValue = eigen_solver.eigenvalues().coeff(1); bigEigenValue = eigen_solver.eigenvalues().coeff(2);
            if(((bigEigenValue/midEigenValue)<5))
                laserConerFeature->push_back(laserCloud->points[ind]);
        }
    }


    // ////////////////////////////////////////////////////////////////////////
    // 提取planar point
    // ////////////////////////////////////////////////////////////////////////
    int smallestPickedNum = 0;
    for (int k = sp; k <= ep; k++)
    {
        int ind = cloudSortInd[k];


        if ((cloudCurvature[ind] < 0.1) && (cloudNeighborPicked[ind] != 1)) {
            cloudLabel[ind] = -1;

            PointType searchPoint = laserCloud->points[ind];
            kdcloud.nearestKSearch(searchPoint, kNearestNeighbor, kNeighborIndex, kNeighborDistance);
            Eigen::Matrix3f covariance_matrix;

            Eigen::Matrix<float,4,1> searchPt;
            searchPt << searchPoint.x, searchPoint.y, searchPoint.z, 1.0;
            pcl::computeCovarianceMatrixNormalized(*laserCloud, kNeighborIndex, searchPt, covariance_matrix);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
            float smallEigenValue, midEigenValue, bigEigenValue;
            smallEigenValue = eigen_solver.eigenvalues().coeff(0); midEigenValue = eigen_solver.eigenvalues().coeff(1); bigEigenValue = eigen_solver.eigenvalues().coeff(2);

            if((smallEigenValue < 0.01) || ((bigEigenValue/midEigenValue)>20))
                laserSurfFeature->push_back(laserCloud->points[ind]);


            // 如果在周围5个点之内的点的距离小于0.02m，就认为周围点被picked了

        }
    }
    // 标签小于等于0，激光点云的点曲率小于阈值0.1 lessflat，planar points多多益善
}

