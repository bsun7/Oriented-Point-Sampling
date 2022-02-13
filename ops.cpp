#include <ros/ros.h>
#include <iostream>
#include <fstream>
//#include <pcl/console/parse.h>
//#include <pcl/filters/extract_indices.h>
//#include <pcl/visualization/pcl_visualizer.h>
//#include <boost/thread/thread.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
//#include <pcl/sample_consensus/ransac.h>
//#include <pcl/sample_consensus/sac_model_plane.h>
//#include <pcl/sample_consensus/sac_model_sphere.h>
#include <algorithm>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

clock_t t[100];

//read 3D points from pcd file
pcl::PointCloud<pcl::PointXYZ>::Ptr ReadPCD(char* filePath)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	// pcl::io library used for load PCD file 
	if (pcl::io::loadPCDFile<pcl::PointXYZ> (filePath, *cloud) == -1)
	{
		PCL_ERROR("Couldn't read file box_cloud.pcd \n");
	}
	return (cloud);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr DeleteNAN(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	std::vector<int> indices;
	pcl::PointCloud<pcl::PointXYZ>::Ptr out;
	pcl::removeNaNFromPointCloud(*cloud, *out, indices);
	return out;
}

//k nearest neighbor search
void PointsKNN(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int pointNum, int maxK, int** indexKNN, double** distKNN)
{
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud (cloud);
	pcl::PointXYZ searchPoint;
	std::vector<int> pointIdxNKNSearch(maxK);
  	std::vector<float> pointNKNSquaredDistance(maxK);
	for(int i=0; i<pointNum; i++)
	{
		searchPoint.x=cloud->points[i].x;
		searchPoint.y=cloud->points[i].y;
		searchPoint.z=cloud->points[i].z;
		if(kdtree.nearestKSearch (searchPoint, maxK, pointIdxNKNSearch, pointNKNSquaredDistance)>0)
		{
			for(int j=0; j<maxK; j++)
			{
				indexKNN[i][j]=pointIdxNKNSearch[j];
				distKNN[i][j]=pointNKNSquaredDistance[j];
			}
			
		}
	}
	
	/*for(int i=0; i<pointNum; i++)
	{
		for(int j=0; j<k; j++)
		{
			cout<<" "<<indexKNN[i][j];
		}
		cout<<endl;
	}
	cout<<endl;
	for(int i=0; i<pointNum; i++)
	{
		for(int j=0; j<k; j++)
		{
			cout<<" "<<distKNN[i][j];
		}
		cout<<endl;
	}*/
}

//compute sigma and NWR matrix, then get the normals of the points
cv::Mat NormalEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int pointNum, int maxK, int k, int** indexKNN, double** distKNN, double randRate, std::vector<int> &randPoints)
{
	cv::Mat diffM(3, 1, CV_64FC1);
	cv::Mat referP(3, 1, CV_64FC1);
	cv::Mat neighP(3, 1, CV_64FC1);
	cv::Mat nwrSample(3, 3, CV_64FC1);
	cv::Mat nwrMat(3, 3, CV_64FC1);
	cv::Mat normals(pointNum*randRate, 3, CV_64FC1);
	double sigma=0.2;
	//compute sigma
	int nn;    //number of nonzero distance
	/*for(int i=0; i<pointNum; i++)
	{
		nn=0;
		for(int j=0; j<maxK; j++)
		{
			if(distKNN[i][j]!=0)
			{
				nn++;
			}
			if(nn==k)
			{
				sigma=distKNN[i][j]+sigma;
				break;
			}
		}
	}
	sigma=sigma/pointNum;*/
	//compute NWR Matrix
	t[10]=clock();
	//randomly pick some points 
	//std::vector<int> randPoints;
	for(int i=0; i<pointNum; i++)
	{
		randPoints.push_back(i);
	}
	std::random_shuffle(randPoints.begin(), randPoints.end());
	pointNum=pointNum*randRate;
	//cout<<"randRate: "<<randRate<<endl;
	//cout<<"pointNum: "<<pointNum<<endl;
	for(int i=0; i<pointNum; i++)
	{
		referP.at<double>(0,0)=cloud->points[randPoints[i]].x;
		referP.at<double>(1,0)=cloud->points[randPoints[i]].y;
		referP.at<double>(2,0)=cloud->points[randPoints[i]].z;
		nwrMat=cv::Mat::zeros(cv::Size(3,3),CV_64FC1);
		nn=0;
		for(int j=0; j<maxK; j++)
		{
			if(distKNN[randPoints[i]][j]!=0)
			{
				neighP.at<double>(0,0)=cloud->points[indexKNN[randPoints[i]][j]].x;
				neighP.at<double>(1,0)=cloud->points[indexKNN[randPoints[i]][j]].y;
				neighP.at<double>(2,0)=cloud->points[indexKNN[randPoints[i]][j]].z;
				diffM=neighP-referP;
				nwrSample=diffM*diffM.t();
				nwrSample=nwrSample/(distKNN[randPoints[i]][j]*distKNN[randPoints[i]][j]);
				nwrSample=nwrSample*exp(-(distKNN[randPoints[i]][j]*distKNN[randPoints[i]][j])/(2*sigma*sigma));
				nwrMat=nwrMat+nwrSample;
				nn++;
			}
			if(nn==k)
			{
				break;
			}
			
		}
		//compute eigen value
		cv::Mat eigenvalue, eigenvector;
		cv::eigen(nwrMat, eigenvalue, eigenvector);
		//cout<<endl;
		//cout<<nwrMat<<endl;
		//cout<<endl;
		//cout<<eigenvalue<<endl;
		//cout<<endl;
		//cout<<eigenvector<<endl;
		normals.at<double>(i,0)=eigenvector.at<double>(2,0);
		normals.at<double>(i,1)=eigenvector.at<double>(2,1);
		normals.at<double>(i,2)=eigenvector.at<double>(2,2);
	}
	t[11]=clock();
	//printf("NormalEstimation: %lf s\n",(double)(t[11]-t[10])/CLOCKS_PER_SEC);
	return (normals);
}

/*//Open 3D viewer and add point cloud
boost::shared_ptr<pcl::visualization::PCLVisualizer>
simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  //viewer->addCoordinateSystem (1.0, "global");
  viewer->initCameraParameters ();
  return (viewer);
}*/

//implement one point ransac
cv::Mat OnePointRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hori, cv::Mat horiPoints, int maxIteration, double p, double disThreshold, int fitNum, std::vector<int> &best_inliers, std::vector<int> isExist, std::vector<int> randPoints)
{
	int iter=0;
	//plane which ransac find: planeModel is the index of the point on the plane and the plane normal
	cv::Mat planeModel = cv::Mat::zeros(2, 3, CV_64FC1); //The first row is the point on the plane, the second row is the normal of the plane
	int inNum=0;
	std::vector<int> inliers;
	int randInd;
	double e;
	while(iter<maxIteration)
	{
		//randomly select one point from horizontal points
		//t[12]=clock();
		randInd=rand()%cloud_hori->points.size();
		inliers.clear();
		for(int i=0; i<cloud->points.size(); i++)
		{
			if(i==randPoints[randInd] || isExist[i]==0)
			{
				continue;
			}
			//compute the distance between every point to the plane
			double x=cloud->points[i].x-cloud_hori->points[randInd].x;
			double y=cloud->points[i].y-cloud_hori->points[randInd].y;
			double z=cloud->points[i].z-cloud_hori->points[randInd].z;
			double xn=horiPoints.at<double>(randInd,0);
			double yn=horiPoints.at<double>(randInd,1);
			double zn=horiPoints.at<double>(randInd,2);
			double norm_n=sqrt(xn*xn+yn*yn+zn*zn);
			double dis=abs(x*xn+y*yn+z*zn)/norm_n;
			if (dis<disThreshold)
			{
				inliers.push_back(i);
			}
		}
		if (inliers.size()>fitNum)
		{
			inliers.push_back(randPoints[randInd]);
			//cout<<"inliers number: "<<inliers.size()<<endl;
			//e=1-(double(inliers.size())/cloud->points.size());
			//cout<<"e: "<<e<<endl;
			//maxIteration=log(1-p)/log(1-(1-e));
			if (inliers.size()>inNum)
			{
				inNum=inliers.size();
				best_inliers=inliers;
				cout<<"inliers number: "<<inliers.size()<<endl;
				e=1-(double(inliers.size())/cloud->points.size());
				cout<<"e: "<<e<<endl;
				maxIteration=log(1-p)/log(1-(1-e));
			}
		}
		//t[13]=clock();
		//printf("OnePointRANSAC one iteration time: %lf s\n",(double)(t[13]-t[12])/CLOCKS_PER_SEC);
		iter++;
	}
	cout<<"iteration times: "<<iter<<endl;
	cout<<"maxIteration: "<<maxIteration<<endl;
	//if do not find any plane
	if (inNum==0)
	{
		return planeModel;
	}
	//re-estimate the plane using all the inliers points
	cv::Mat pointsInlier(best_inliers.size(), 3,  CV_64FC1);
	for(int i=0; i<best_inliers.size(); i++)
	{
		pointsInlier.at<double>(i,0)=cloud->points[best_inliers[i]].x;
		pointsInlier.at<double>(i,1)=cloud->points[best_inliers[i]].y;
		pointsInlier.at<double>(i,2)=cloud->points[best_inliers[i]].z;
	}
	cv::Mat u,w,v;
	SVD::compute(pointsInlier, w, u, v);
	//cout<<"w: "<<endl<<w<<endl;
	//cout<<"u: "<<endl<<u<<endl;
	//cout<<"v: "<<endl<<v.row(2)<<endl;
	//cout<<"mean of v: "<<endl<<mean(v)[0]<<endl;
	planeModel.at<double>(1,0)=v.at<double>(2,0);
	planeModel.at<double>(1,1)=v.at<double>(2,1);
	planeModel.at<double>(1,2)=v.at<double>(2,2);

	double colMean;
	cv::Mat col(best_inliers.size(), 1,  CV_64FC1);
	for(int i=0; i<3; i++)
	{
		col=pointsInlier.col(i);
		colMean=mean(col)[0];
		planeModel.at<double>(0,i)=colMean;
	}
	return planeModel;
}

int main(int argc, char **argv)
{
	srand((unsigned)time(NULL));
	t[1]=clock();
	// init a node named 'read_points'
	ros::init(argc, argv, "read_points");  
	ros::NodeHandle n;
	
	//read 3D points from pcd file
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZ>);
	cloud1 = ReadPCD("/home/victor/project_ws/src/ground_plane/data/pcd_night/top1.pcd");
	/*//write all the points in cloud to ply file
	ofstream myfile2("/home/victor/project_ws/src/ground_plane/src/allPoints_origin.ply", ios::out);
	if(!myfile2)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		myfile2<<"ply"<<endl;
		myfile2<<"format ascii 1.0"<<endl;
		myfile2<<"element vertex "<<cloud1->points.size()<<endl;
		myfile2<<"property double x"<<endl;
		myfile2<<"property double y"<<endl;
		myfile2<<"property double z"<<endl;
		myfile2<<"property uchar red"<<endl;
		myfile2<<"property uchar green"<<endl;
		myfile2<<"property uchar blue"<<endl;
		myfile2<<"element face 0"<<endl;
		myfile2<<"end_header"<<endl;
		for(int i=0; i<cloud1->points.size(); i++)
		{
			myfile2<<cloud1->points[i].x<<" "<<cloud1->points[i].y<<" "<<cloud1->points[i].z<<" "<<"255 0 0"<<endl;
		}
		myfile2.close();
	}*/


	//delete the nan points
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	cloud = DeleteNAN(cloud1);
	t[2]=clock();
	printf("%lf s\n",(double)(t[2]-t[1])/CLOCKS_PER_SEC);
	int pointNum=cloud->points.size();
	//int pointNum=30;
	//cout the 3d points
	/*cout << "loaded " << cloud->width * cloud->height << " data points from box_cloud.pcd with the following fields: " << endl;
	for(int i=0; i<pointNum; i++)
	{
		cout<<" "<<cloud->points[i].x<<" "<<cloud->points[i].y<<" "<<cloud->points[i].z<<endl;
	}*/
	
	//write all the points in cloud to ply file
	ofstream myfile1("/home/victor/project_ws/src/ground_plane/src/allPoints.ply", ios::out);
	if(!myfile1)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		myfile1<<"ply"<<endl;
		myfile1<<"format ascii 1.0"<<endl;
		myfile1<<"element vertex "<<pointNum<<endl;
		myfile1<<"property double x"<<endl;
		myfile1<<"property double y"<<endl;
		myfile1<<"property double z"<<endl;
		myfile1<<"property uchar red"<<endl;
		myfile1<<"property uchar green"<<endl;
		myfile1<<"property uchar blue"<<endl;
		myfile1<<"element face 0"<<endl;
		myfile1<<"end_header"<<endl;
		for(int i=0; i<pointNum; i++)
		{
			myfile1<<cloud->points[i].x<<" "<<cloud->points[i].y<<" "<<cloud->points[i].z<<" "<<"255 0 0"<<endl;
		}
		myfile1.close();
	}

	t[3]=clock();
	printf("%lf s\n",(double)(t[3]-t[2])/CLOCKS_PER_SEC);
	//k nearest neighbor search
	int maxK=30, k=30;
  	int** indexKNN;
  	indexKNN=new int*[pointNum];
  	for(int i=0; i<pointNum; i++)
  	{
  		indexKNN[i]=new int[maxK];
	}
	double** distKNN;
	distKNN=new double*[pointNum];
	for(int i=0; i<pointNum; i++)
  	{
  		distKNN[i]=new double[maxK];
	}
	PointsKNN(cloud, pointNum, maxK, indexKNN, distKNN);

	t[4]=clock();
	printf("%lf s\n",(double)(t[4]-t[3])/CLOCKS_PER_SEC);

	/*cout<<endl;
	cout<<" "<<cloud->points[0].x<<" "<<cloud->points[0].y<<" "<<cloud->points[0].z<<endl;
	cout<<" "<<cloud->points[30000].x<<" "<<cloud->points[30000].y<<" "<<cloud->points[30000].z<<endl;
	cout<<" "<<cloud->points[10000].x<<" "<<cloud->points[10000].y<<" "<<cloud->points[10000].z<<endl;*/
	
	//normal estimation
	//cv::Mat normals(pointNum, 3, CV_64FC1);
	cv::Mat normals;
	std::vector<int> randPoints;
	double randRate=0.1;
	normals=NormalEstimation(cloud, pointNum, maxK, k, indexKNN, distKNN, randRate, randPoints);
	/*for(int i=0; i<normals.rows; i++)
	{
		if(normals.at<double>(i,0)==0 && normals.at<double>(i,1)==0 && normals.at<double>(i,2)==0)
		{
			cout<<endl;
			cout<<"i="<<i<<endl;
			cout<<normals.at<double>(i,0)<<" "<<normals.at<double>(i,1)<<" "<<normals.at<double>(i,2)<<endl;
		}
	}*/
	cout<<endl;
	cout<<"Number of normals: "<<normals.rows<<endl;
	cout<<"Number of points: "<<cloud->points.size()<<endl;

	t[5]=clock();
	printf("%lf s\n",(double)(t[5]-t[4])/CLOCKS_PER_SEC);

	//cout<<normals<<endl;
	
	//use kmeans to find the planes
	/*cv::Mat labels, centers;
	cv::Mat normals_32;
	normals.convertTo(normals_32, CV_32FC1);
	kmeans(normals_32, 3, labels, TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 1000, 0.1), 3, KMEANS_PP_CENTERS, centers);
	cout<<endl;
	cout<<labels<<endl;
	cout<<endl;
	cout<<centers<<endl;*/

	//find the points in horizontal plane
	//int normals_size=1;
	double ab, an, bn, cosr, interAngle;
	cv::Mat horiPoints;  //(1, 3, CV_64FC1); //the normals of the horizontal plane points 
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hori (new pcl::PointCloud<pcl::PointXYZ>);
	//std::vector<int> indn;  //the index in normals
	for(int i=0; i<normals.rows; i++)
	{
		//compute the angle between normal and yAxis direction (0,1,0)
		ab=normals.at<double>(i,1);
		an=1;
		bn=sqrt(normals.at<double>(i,0)*normals.at<double>(i,0)+normals.at<double>(i,1)*normals.at<double>(i,1)+normals.at<double>(i,2)*normals.at<double>(i,2));
		cosr=ab/(an*bn);
		//cout<<M_PI<<endl;
		//cout<<acos(cosr)*180/M_PI<<endl;
		interAngle=acos(cosr)*180/M_PI;
		/*if(isnan(interAngle))
		{
			cout<<"i="<<i<<endl;
			cout<<normals.at<double>(i,0)<<" "<<normals.at<double>(i,1)<<" "<<normals.at<double>(i,2)<<endl;
		}*/
		if(interAngle<=7 || interAngle>=173)
		{
			cv::Mat hpRow(1, 3, CV_64FC1);
			hpRow.at<double>(0,0)=normals.at<double>(i,0);
			hpRow.at<double>(0,1)=normals.at<double>(i,1);
			hpRow.at<double>(0,2)=normals.at<double>(i,2);
			cloud_hori->push_back(cloud->points[randPoints[i]]);
			//indn.push_back(i);
			//cout<<endl;
			//cout<<hpRow<<endl;
			horiPoints.push_back(hpRow);
		}
	}
	cout<<endl;
	cout<<horiPoints.rows<<endl;
	//cout<<horiPoints.row(0)<<endl;
	//cout<<horiPoints.row(1)<<endl;
	cout<<cloud_hori->points.size()<<endl;
	//cout<<indn.size()<<endl;
	//cout<<horiPoints<<endl;

	t[6]=clock();
	printf("%lf s\n",(double)(t[6]-t[5])/CLOCKS_PER_SEC);

	//write the horizontal plane into ply file
	ofstream myfile("/home/victor/project_ws/src/ground_plane/src/horiPoints.ply", ios::out);
	if(!myfile)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		myfile<<"ply"<<endl;
		myfile<<"format ascii 1.0"<<endl;
		myfile<<"element vertex "<<horiPoints.rows<<endl;
		myfile<<"property double x"<<endl;
		myfile<<"property double y"<<endl;
		myfile<<"property double z"<<endl;
		myfile<<"property uchar red"<<endl;
		myfile<<"property uchar green"<<endl;
		myfile<<"property uchar blue"<<endl;
		myfile<<"element face 0"<<endl;
		myfile<<"end_header"<<endl;
		for(int i=0; i<cloud_hori->points.size(); i++)
		{
			myfile<<cloud_hori->points[i].x<<" "<<cloud_hori->points[i].y<<" "<<cloud_hori->points[i].z<<" "<<"255 0 0"<<endl;
		}
		myfile.close();
	}

	t[7]=clock();
	printf("%lf s\n",(double)(t[7]-t[6])/CLOCKS_PER_SEC);

	//use one point ransac to find horizontal plane
	std::vector<std::vector<int> > inliers;
	std::vector<cv::Mat> planeModel;
	int minNum=10000;  //the minimum number of points in the plane
	//std::vector<int> isExist(cloud_hori->points.size(), 1);
	std::vector<int> isExist(cloud->points.size(), 1);
	
	while(accumulate(isExist.begin(), isExist.end(), 0)>minNum)
	{
		cv::Mat plane;  //first row is the point on the plane, second row is the normal
		std::vector<int> inl;
		//t[10]=clock();
		cout<<"isExist number: "<<endl<<accumulate(isExist.begin(), isExist.end(), 0)<<endl;
		//t[11]=clock();
		//printf("%lf s\n",(double)(t[11]-t[10])/CLOCKS_PER_SEC);
		t[10]=clock();
		plane=OnePointRANSAC(cloud, cloud_hori, horiPoints, horiPoints.rows, 0.99, 0.05, minNum, inl, isExist, randPoints);
		t[11]=clock();
		printf("OnePointRANSAC time: %lf s\n",(double)(t[11]-t[10])/CLOCKS_PER_SEC);
		if (countNonZero(plane)!=0)
		{
			planeModel.push_back(plane);
			inliers.push_back(inl);
			for(int i=0; i<inl.size(); i++)
			{
				isExist[inl[i]]=0;
			}
		}
		else
		{
			break;
		}
	}
	cout<<"Before aggregate planeModel size: "<<endl<<planeModel.size()<<endl;
	//aggregate the planes which should be the same one
	std::vector<std::vector<int> > inliers_fi;
	std::vector<cv::Mat> planeModel_fi;
	std::vector<int> indPlane;
	double disPlane;
	for(int i=0; i<planeModel.size(); i++)
	{
		indPlane.push_back(i);
	}
	for(int i=0; i<planeModel.size(); i++)
	{
		for(int j=i+1; j<planeModel.size(); j++)
		{
			disPlane=sqrt((planeModel[i].at<double>(0,0)-planeModel[j].at<double>(0,0))*(planeModel[i].at<double>(0,0)-planeModel[j].at<double>(0,0))+(planeModel[i].at<double>(0,1)-planeModel[j].at<double>(0,1))*(planeModel[i].at<double>(0,1)-planeModel[j].at<double>(0,1))+(planeModel[i].at<double>(0,2)-planeModel[j].at<double>(0,2))*(planeModel[i].at<double>(0,2)-planeModel[j].at<double>(0,2)));
			cout<<"i="<<i<<" j="<<j<<" distance: "<<disPlane<<endl;
			if (disPlane<0.01)
			{
				indPlane[j]=i;
			}
		}
	}
	for(int i=0; i<indPlane.size(); i++)
	{
		if(indPlane[i]==-1)
		{
			continue;
		}
		std::vector<int> vecfi;
		vecfi.insert(vecfi.end(), inliers[i].begin(), inliers[i].end());
		cv::Mat pm(2, 3, CV_64FC1);
		pm.at<double>(0,0)=planeModel[i].at<double>(0,0);
		pm.at<double>(0,1)=planeModel[i].at<double>(0,1);
		pm.at<double>(0,2)=planeModel[i].at<double>(0,2);
		int n=1;
		//pm.at<double>(1,0)=planeModel[i].at<double>(1,0);
		//pm.at<double>(1,1)=planeModel[i].at<double>(1,1);
		//pm.at<double>(1,2)=planeModel[i].at<double>(1,2);
		for(int j=i+1; j<indPlane.size(); j++)
		{
			if (indPlane[j]==indPlane[i])
			{
				pm.at<double>(0,0)=pm.at<double>(0,0)+planeModel[j].at<double>(0,0);
				pm.at<double>(0,1)=pm.at<double>(0,1)+planeModel[j].at<double>(0,1);
				pm.at<double>(0,2)=pm.at<double>(0,2)+planeModel[j].at<double>(0,2);
				n++;
				vecfi.insert(vecfi.end(), inliers[j].begin(), inliers[j].end());
				indPlane[j]=-1;
			}
		}
		pm.at<double>(0,0)=pm.at<double>(0,0)/n;
		pm.at<double>(0,1)=pm.at<double>(0,1)/n;
		pm.at<double>(0,2)=pm.at<double>(0,2)/n;
		cv::Mat pointsIn(vecfi.size(), 3, CV_64FC1);
		for(int j=0; j<vecfi.size(); j++)
		{
			pointsIn.at<double>(j,0)=cloud->points[vecfi[j]].x;
			pointsIn.at<double>(j,1)=cloud->points[vecfi[j]].y;
			pointsIn.at<double>(j,2)=cloud->points[vecfi[j]].z;
		}
		cv::Mat u0,w0,v0;
		SVD::compute(pointsIn, w0, u0, v0);
		pm.at<double>(1,0)=v0.at<double>(2,0);
		pm.at<double>(1,1)=v0.at<double>(2,1);
		pm.at<double>(1,2)=v0.at<double>(2,2);
		planeModel_fi.push_back(pm);
		inliers_fi.push_back(vecfi);
		indPlane[i]=-1;
	}
	cout<<"After aggregate planeModel size: "<<endl<<planeModel_fi.size()<<endl;
	t[8]=clock();
	printf("%lf s\n",(double)(t[8]-t[7])/CLOCKS_PER_SEC);

	//write the horizontal plane into ply file
	for(int i=0; i<inliers_fi.size(); i++)
	{
		cout<<"inliers size: "<<endl<<inliers_fi[i].size()<<endl;
		char outPath[200];
		sprintf(outPath, "/home/victor/project_ws/src/ground_plane/src/ransacPlane%d.ply", i);
		ofstream myfile0(outPath, ios::out);
		if(!myfile0)
		{
			cout<<"error!"<<endl;
		}
		else
		{
			myfile0<<"ply"<<endl;
			myfile0<<"format ascii 1.0"<<endl;
			myfile0<<"element vertex "<<inliers_fi[i].size()<<endl;
			myfile0<<"property double x"<<endl;
			myfile0<<"property double y"<<endl;
			myfile0<<"property double z"<<endl;
			myfile0<<"property uchar red"<<endl;
			myfile0<<"property uchar green"<<endl;
			myfile0<<"property uchar blue"<<endl;
			myfile0<<"element face 0"<<endl;
			myfile0<<"end_header"<<endl;
			for(int j=0; j<inliers_fi[i].size(); j++)
			{
				myfile0<<cloud->points[inliers_fi[i][j]].x<<" "<<cloud->points[inliers_fi[i][j]].y<<" "<<cloud->points[inliers_fi[i][j]].z<<" "<<"255 0 0"<<endl;
			}
			myfile0.close();
		}
	}
	
	t[9]=clock();
	printf("%lf s\n",(double)(t[9]-t[8])/CLOCKS_PER_SEC);


	/*//use ransac in pcl to find horizontal plane
	std::vector<int> inliers;
	pcl::PointCloud<pcl::PointXYZ>::Ptr final (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud_hori));
	pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_p);
	ransac.setDistanceThreshold (.01);
	ransac.computeModel();
	ransac.getInliers(inliers);
	//copies all inliers of the model computed to another PointCloud
  	pcl::copyPointCloud<pcl::PointXYZ>(*cloud_hori, inliers, *final);
	//visualize the inliers
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	viewer = simpleVis(final);
	while (!viewer->wasStopped ())
  	{
    	viewer->spinOnce (100);
    	boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  	}*/
	return 0;
}

