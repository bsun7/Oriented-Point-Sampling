#include "pcltopcl2.h"

using namespace std;

bool convertPointCloudToPointCloud2(const sensor_msgs::PointCloudConstPtr &cloudin, sensor_msgs::PointCloud2Ptr &PointCloud2)
{
    if (sensor_msgs::convertPointCloudToPointCloud2(*cloudin, *PointCloud2))
        return true;
    return false;
}

bool convertPointCloud2ToPCLXYZ(const sensor_msgs::PointCloud2ConstPtr &cloudin, pcl::PointCloud<pcl::PointXYZ>::Ptr &pclPointcloud)
{
  pcl::fromROSMsg(*cloudin, *pclPointcloud);
  return true;
}

void convertPointCloudToPCLXYZ(const sensor_msgs::PointCloudConstPtr &cloudin)
{   
    sensor_msgs::PointCloud2Ptr PointCloud2;
    convertPointCloudToPointCloud2(cloudin, PointCloud2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclPointcloud;
    convertPointCloud2ToPCLXYZ(PointCloud2, pclPointcloud);
    ROS_INFO("pclPointCloud converted successfully");
    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pclTopcl2");  
	ros::NodeHandle n;
    /* define a subscriber named _subCloud, and subscribe topic '/sensor_msg::PointCloud'*/
    ros::Subscriber _subCloud;
    ros::Publisher _pubCloudPCLXYZ;
    _subCloud = n.subscribe<sensor_msgs::PointCloud2>
      ("/sensor_msg::PointCloud", 1, convertPointCloudToPCLXYZ);
    ros::spin();
    return 0;
}