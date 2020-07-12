#include <ros/ros.h>
#include <iostream>
#include <fstream>
//#include <pcl/console/parse.h>
//#include <pcl/filters/extract_indices.h>
//#include <pcl/visualization/pcl_visualizer.h>
//#include <boost/thread/thread.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
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
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>


using namespace cv;
using namespace std;

clock_t t[100];

//read 3D points from pcd file
pcl::PointCloud<pcl::PointXYZ>::Ptr ReadPCD(char* filePath)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ> (filePath, *cloud) == -1)
	{
		PCL_ERROR("Couldn't read file box_cloud.pcd \n");
		
	}
	return cloud;
}

//read 3D points from ply file
pcl::PointCloud<pcl::PointXYZ>::Ptr ReadPLY(char* filePath)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PLYReader Reader;
	Reader.read(filePath, *cloud);
	return cloud;
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
cv::Mat NormalEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int pointNum, int maxK, int** indexKNN, double** distKNN)
{
	cv::Mat diffM(3, 1, CV_64FC1);
	cv::Mat referP(3, 1, CV_64FC1);
	cv::Mat neighP(3, 1, CV_64FC1);
	cv::Mat nwrSample(3, 3, CV_64FC1);
	cv::Mat nwrMat(3, 3, CV_64FC1);
	cv::Mat normals(pointNum, 3, CV_64FC1);
	double sigma=0.2;
	//compute sigma
	//int nn;    //number of nonzero distance
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
	/*for(int i=0; i<pointNum; i++)
	{
		randPoints.push_back(i);
	}
	std::random_shuffle(randPoints.begin(), randPoints.end());
	pointNum=pointNum*randRate;*/
	//cout<<"randRate: "<<randRate<<endl;
	//cout<<"pointNum: "<<pointNum<<endl;
	for(int i=0; i<pointNum; i++)
	{
		referP.at<double>(0,0)=cloud->points[i].x;
		referP.at<double>(1,0)=cloud->points[i].y;
		referP.at<double>(2,0)=cloud->points[i].z;
		nwrMat=cv::Mat::zeros(cv::Size(3,3),CV_64FC1);
		//nn=0;
		for(int j=0; j<maxK; j++)
		{
			if(distKNN[i][j]!=0)
			{
				neighP.at<double>(0,0)=cloud->points[indexKNN[i][j]].x;
				neighP.at<double>(1,0)=cloud->points[indexKNN[i][j]].y;
				neighP.at<double>(2,0)=cloud->points[indexKNN[i][j]].z;
				diffM=neighP-referP;
				nwrSample=diffM*diffM.t();
				nwrSample=nwrSample/(distKNN[i][j]*distKNN[i][j]);
				nwrSample=nwrSample*exp(-(distKNN[i][j]*distKNN[i][j])/(2*sigma*sigma));
				nwrMat=nwrMat+nwrSample;
				//nn++;
			}
			/*if(nn==k)
			{
				break;
			}*/
			
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
	return normals;
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
cv::Mat OnePointRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hori, cv::Mat horiPoints, int maxIteration, double p, double disThreshold, int fitNum, std::vector<int> &best_inliers, std::vector<int> isExist)
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
		for(int i=0; i<cloud_hori->points.size(); i++)
		{
			if(i==randInd || isExist[i]==0)
			{
				continue;
			}
			//compute the distance between every point to the plane
			double x=cloud_hori->points[i].x-cloud_hori->points[randInd].x;
			double y=cloud_hori->points[i].y-cloud_hori->points[randInd].y;
			double z=cloud_hori->points[i].z-cloud_hori->points[randInd].z;
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
			inliers.push_back(randInd);
			//cout<<"inliers number: "<<inliers.size()<<endl;
			//e=1-(double(inliers.size())/cloud->points.size());
			//cout<<"e: "<<e<<endl;
			//maxIteration=log(1-p)/log(1-(1-e));
			if (inliers.size()>inNum)
			{
				inNum=inliers.size();
				best_inliers=inliers;
				cout<<"inliers number: "<<inliers.size()<<endl;
				e=1-(double(inliers.size())/cloud_hori->points.size());
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
	double colSum[3]={0,0,0};
	for(int i=0; i<best_inliers.size(); i++)
	{
		pointsInlier.at<double>(i,0)=cloud_hori->points[best_inliers[i]].x;
		pointsInlier.at<double>(i,1)=cloud_hori->points[best_inliers[i]].y;
		pointsInlier.at<double>(i,2)=cloud_hori->points[best_inliers[i]].z;
		colSum[0]+=cloud_hori->points[best_inliers[i]].x;
		colSum[1]+=cloud_hori->points[best_inliers[i]].y;
		colSum[2]+=cloud_hori->points[best_inliers[i]].z;
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

	for(int i=0; i<3; i++)
	{
		planeModel.at<double>(0,i)=colSum[i]/best_inliers.size();
	}
	return planeModel;
}
//read point cloud from inputFile and find horizontal planes, then write results to outputFile
void FindHorizontalPlanes(char* inputFile, char* outputFile)
{
	//srand((unsigned)time(NULL));
	t[1]=clock();
	//ros::init(argc, argv, "read_points");
	//ros::NodeHandle n;
	//read 3D points from pcd file
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZ>);
	//cloud1 = ReadPCD("/home/victor/project_ws/src/ground_plane/data/home_data_ascii/scene1_ascii.pcd");
    cloud1 = ReadPCD(inputFile);
	//cloud1 = ReadPLY(inputFile);
	t[2]=clock();
	printf("%lf s\n",(double)(t[2]-t[1])/CLOCKS_PER_SEC);
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
	t[3]=clock();
	printf("%lf s\n",(double)(t[3]-t[2])/CLOCKS_PER_SEC);
	int pointNum=cloud->points.size();
	//int pointNum=30;
	//cout the 3d points
	/*cout << "loaded " << cloud->width * cloud->height << " data points from box_cloud.pcd with the following fields: " << endl;
	for(int i=0; i<pointNum; i++)
	{
		cout<<" "<<cloud->points[i].x<<" "<<cloud->points[i].y<<" "<<cloud->points[i].z<<endl;
	}*/
	
	//write all the points in cloud to ply file
    char mkFile[100];
    sprintf(mkFile, "/home/victor/project_ws/src/ground_plane/src/%s/", outputFile);
    cout<<mkFile<<endl;
    int isCreate=mkdir(mkFile, S_IRWXU);
    if(!isCreate)
    {
        printf("Create Path:%s\n",mkFile);
    }
    else
    {
        printf("Create Path Failed!\n");
        return;
    }
    char outFile1[100];
    sprintf(outFile1, "/home/victor/project_ws/src/ground_plane/src/%s/allPoints.ply", outputFile);
	ofstream myfile1(outFile1, ios::out);
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

	t[4]=clock();
	printf("%lf s\n",(double)(t[4]-t[3])/CLOCKS_PER_SEC);
	//randomly pick some points from all the points
	/*std::vector<int> randPoints;
	double randRate=0.1;
	for(int i=0; i<pointNum; i++)
	{
		randPoints.push_back(i);
	}
	std::random_shuffle(randPoints.begin(), randPoints.end());
	pointNum=pointNum*randRate;*/
	//k nearest neighbor search
	int maxK=40;
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

	t[5]=clock();
	printf("%lf s\n",(double)(t[5]-t[4])/CLOCKS_PER_SEC);

	/*cout<<endl;
	cout<<" "<<cloud->points[0].x<<" "<<cloud->points[0].y<<" "<<cloud->points[0].z<<endl;
	cout<<" "<<cloud->points[30000].x<<" "<<cloud->points[30000].y<<" "<<cloud->points[30000].z<<endl;
	cout<<" "<<cloud->points[10000].x<<" "<<cloud->points[10000].y<<" "<<cloud->points[10000].z<<endl;*/
	
	//normal estimation
	//cv::Mat normals(pointNum, 3, CV_64FC1);
	cv::Mat normals;
	normals=NormalEstimation(cloud, pointNum, maxK, indexKNN, distKNN);

	cout<<endl;
	cout<<"Number of normals: "<<normals.rows<<endl;
	cout<<"Number of points: "<<cloud->points.size()<<endl;

	t[6]=clock();
	printf("%lf s\n",(double)(t[6]-t[5])/CLOCKS_PER_SEC);

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
	std::vector<int> hori_ind; //the index in cloud for the horizontal points
	for(int i=0; i<normals.rows; i++)
	{
		//compute the angle between normal and yAxis direction (0,1,0)/zAxis direction (0,0,1)
		ab=normals.at<double>(i,2);
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
			cloud_hori->push_back(cloud->points[i]);
			hori_ind.push_back(i);
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

	t[7]=clock();
	printf("%lf s\n",(double)(t[7]-t[6])/CLOCKS_PER_SEC);

	/*//write the normals into txt file
	char outFile3[100];
    sprintf(outFile3, "/home/victor/project_ws/src/ground_plane/src/%s/normals.txt", outputFile);
	ofstream myfile3(outFile3, ios::out);
	if(!myfile3)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		for(int i=0; i<horiPoints.rows; i++)
		{
			myfile3<<horiPoints.at<double>(i,0)<<" "<<horiPoints.at<double>(i,1)<<" "<<horiPoints.at<double>(i,2)<<endl;
		}
		myfile3.close();
	}*/

	//write the horizontal plane into ply file
    char outFile2[100];
    sprintf(outFile2, "/home/victor/project_ws/src/ground_plane/src/%s/horiPoints.ply", outputFile);
	ofstream myfile(outFile2, ios::out);
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

	t[8]=clock();
	printf("%lf s\n",(double)(t[8]-t[7])/CLOCKS_PER_SEC);

	//use one point ransac to find horizontal plane
	std::vector<std::vector<int> > inliers;
	std::vector<cv::Mat> planeModel;
	int minNum=20;  //the minimum number of points in the plane
	double dis_thresh=0.01;  //threshold of the distance between point and plane or plane and plane 
	std::vector<int> isExist(cloud_hori->points.size(), 1);
	//std::vector<int> isExist(cloud->points.size(), 1);
	
	while(accumulate(isExist.begin(), isExist.end(), 0)>minNum)
	{
		cv::Mat plane;  //first row is the point on the plane, second row is the normal
		std::vector<int> inl;
		//t[10]=clock();
		cout<<"isExist number: "<<endl<<accumulate(isExist.begin(), isExist.end(), 0)<<endl;
		//t[11]=clock();
		//printf("%lf s\n",(double)(t[11]-t[10])/CLOCKS_PER_SEC);
		//t[10]=clock();
		plane=OnePointRANSAC(cloud_hori, horiPoints, horiPoints.rows, 0.99, dis_thresh, minNum, inl, isExist);
		//t[11]=clock();
		//printf("OnePointRANSAC time: %lf s\n",(double)(t[11]-t[10])/CLOCKS_PER_SEC);
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
	cout<<"planeModel size: "<<endl<<planeModel.size()<<endl;
	//aggregate the planes which should be the same one
	/*std::vector<std::vector<int> > inliers_fi;
	std::vector<cv::Mat> planeModel_fi;
	std::vector<int> indPlane;
	double disPlane;
	for(int i=0; i<planeModel.size(); i++)
	{
		indPlane.push_back(i);
	}
	for(int i=0; i<planeModel.size(); i++)
	{
		if (indPlane[i]!=i)
			continue;
		for(int j=i+1; j<planeModel.size(); j++)
		{
			if (indPlane[j]!=j)
				continue;
			//compute the offset of plane i
			double xn_i=planeModel[i].at<double>(1,0);
			double yn_i=planeModel[i].at<double>(1,1);
			double zn_i=planeModel[i].at<double>(1,2);
			double x_i=planeModel[i].at<double>(0,0);
			double y_i=planeModel[i].at<double>(0,1);
			double z_i=planeModel[i].at<double>(0,2);
			double os_i=-(xn_i*x_i+yn_i*y_i+zn_i*z_i);

			//compute the offset of plane j
			double xn_j=planeModel[j].at<double>(1,0);
			double yn_j=planeModel[j].at<double>(1,1);
			double zn_j=planeModel[j].at<double>(1,2);
			double x_j=planeModel[j].at<double>(0,0);
			double y_j=planeModel[j].at<double>(0,1);
			double z_j=planeModel[j].at<double>(0,2);
			double os_j=-(xn_j*x_j+yn_j*y_j+zn_j*z_j);

			disPlane=abs(planeModel[i].at<double>(0,2)-planeModel[j].at<double>(0,2));
			cout<<"i="<<i<<" j="<<j<<" distance: "<<disPlane<<endl;
			if (disPlane<dis_thresh)
			{
				indPlane[j]=indPlane[i];				
			}
		}
	}
	for(int i=0; i<indPlane.size(); i++)
	{
		cout<<"indPlane["<<i<<"]= "<<indPlane[i]<<endl;
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
		pm.at<double>(0,0)=planeModel[i].at<double>(0,0)*inliers[i].size();
		pm.at<double>(0,1)=planeModel[i].at<double>(0,1)*inliers[i].size();
		pm.at<double>(0,2)=planeModel[i].at<double>(0,2)*inliers[i].size();
		int n=inliers[i].size();
		//pm.at<double>(1,0)=planeModel[i].at<double>(1,0);
		//pm.at<double>(1,1)=planeModel[i].at<double>(1,1);
		//pm.at<double>(1,2)=planeModel[i].at<double>(1,2);
		for(int j=i+1; j<indPlane.size(); j++)
		{
			if (indPlane[j]==indPlane[i])
			{
				pm.at<double>(0,0)=pm.at<double>(0,0)+planeModel[j].at<double>(0,0)*inliers[j].size();
				pm.at<double>(0,1)=pm.at<double>(0,1)+planeModel[j].at<double>(0,1)*inliers[j].size();
				pm.at<double>(0,2)=pm.at<double>(0,2)+planeModel[j].at<double>(0,2)*inliers[j].size();
				n+=inliers[j].size();
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
	cout<<"After aggregate planeModel size: "<<endl<<planeModel_fi.size()<<endl;*/
	t[9]=clock();
	printf("%lf s\n", (double)(t[9]-t[8])/CLOCKS_PER_SEC);
	printf("Finding planes time: %lf s\n", (double)(t[9]-t[4])/CLOCKS_PER_SEC);

	//assign all the points in the point cloud to the nearest plane(distance less than threshold) and get final inliers
	//std::vector<std::vector<int> > inliers_final;
	//std::vector<cv::Mat> planeModel_final;
	std::vector<int> isUsed(cloud->points.size(), 0);
	for(int i=0; i<inliers.size(); i++)
	{
		for(int j=0; j<inliers[i].size(); j++)
		{
			inliers[i][j]=hori_ind[inliers[i][j]];//transfer the cloud_hori index to cloud index
			isUsed[inliers[i][j]]=1;
		}
	}

	for(int i=0; i<cloud->points.size(); i++)
	{
		if(isUsed[i]==1)
		{
			continue;
		}
		std::vector<double> dis_in(inliers.size(), 0);
		double min_dis;
		int min_ind;
		for(int j=0; j<inliers.size(); j++)
		{
			//compute the distance between every point to the inliers_fi planes
			//double x=cloud->points[i].x-planeModel_fi[j].at<double>(0,0);
			//double y=cloud->points[i].y-planeModel_fi[j].at<double>(0,1);
			//double z=cloud->points[i].z-planeModel_fi[j].at<double>(0,2);
			//double xn=planeModel_fi[j].at<double>(1,0);
			//double yn=planeModel_fi[j].at<double>(1,1);
			//double zn=planeModel_fi[j].at<double>(1,2);
			//double norm_n=sqrt(xn*xn+yn*yn+zn*zn);
			//dis_in[j]=abs(x*xn+y*yn+z*zn)/norm_n;
			dis_in[j]=abs(cloud->points[i].z-planeModel[j].at<double>(0,2));
			//cout<<"dis_in["<<j<<"]= "<<dis_in[j]<<endl;
			if (j==0)
			{
				min_dis=dis_in[j];
				min_ind=j;
			}
			else
			{
				if (dis_in[j]<min_dis)
				{
					min_dis=dis_in[j];
					min_ind=j;
				}
			}
		}
		if (min_dis<dis_thresh)
		{
			inliers[min_ind].push_back(i);
		}
		isUsed[i]=1;
	}
	//re-compute the normals and center point of the points in inliers_fi
	for(int i=0; i<inliers.size(); i++)
	{
		//cout<<"i= "<<i<<endl;
		cv::Mat pointsInlier(inliers[i].size(), 3,  CV_64FC1);
		double colSum[3]={0,0,0};
		for(int j=0; j<inliers[i].size(); j++)
		{
			//cout<<"j= "<<j<<"/"<<inliers_fi[i].size()<<endl;
			pointsInlier.at<double>(j,0)=cloud->points[inliers[i][j]].x;
			pointsInlier.at<double>(j,1)=cloud->points[inliers[i][j]].y;
			pointsInlier.at<double>(j,2)=cloud->points[inliers[i][j]].z;
			colSum[0]+=pointsInlier.at<double>(j,0);
			colSum[1]+=pointsInlier.at<double>(j,1);
			colSum[2]+=pointsInlier.at<double>(j,2);
		}
		cv::Mat u,w,v;
		SVD::compute(pointsInlier, w, u, v);
		planeModel[i].at<double>(1,0)=v.at<double>(2,0);
		planeModel[i].at<double>(1,1)=v.at<double>(2,1);
		planeModel[i].at<double>(1,2)=v.at<double>(2,2);

		for(int j=0; j<3; j++)
		{
			planeModel[i].at<double>(0,j)=colSum[j]/inliers[i].size();
		}
	}
	t[10]=clock();

	//write the horizontal plane into ply file and store the normals of each plane
	int num_allpoints=0;
	for(int i=0; i<inliers.size(); i++)
	{
		num_allpoints+=inliers[i].size();
		cout<<"inliers size: "<<endl<<inliers[i].size()<<endl;
		//write the horizontal plane into ply file
		char outPath[100];
		sprintf(outPath, "/home/victor/project_ws/src/ground_plane/src/%s/ransacPlane%d.ply", outputFile, i);
		ofstream myfile0(outPath, ios::out);
		if(!myfile0)
		{
			cout<<"error!"<<endl;
		}
		else
		{
			myfile0<<"ply"<<endl;
			myfile0<<"format ascii 1.0"<<endl;
			myfile0<<"element vertex "<<inliers[i].size()<<endl;
			myfile0<<"property double x"<<endl;
			myfile0<<"property double y"<<endl;
			myfile0<<"property double z"<<endl;
			myfile0<<"property uchar red"<<endl;
			myfile0<<"property uchar green"<<endl;
			myfile0<<"property uchar blue"<<endl;
			myfile0<<"element face 0"<<endl;
			myfile0<<"end_header"<<endl;
			for(int j=0; j<inliers[i].size(); j++)
			{
				myfile0<<cloud->points[inliers[i][j]].x<<" "<<cloud->points[inliers[i][j]].y<<" "<<cloud->points[inliers[i][j]].z<<" "<<"255 0 0"<<endl;
			}
			myfile0.close();

		}
		//store the normals and center point of each plane
		char outPath1[100];
		sprintf(outPath1, "/home/victor/project_ws/src/ground_plane/src/%s/normal%d.txt", outputFile, i);
		ofstream myfile2(outPath1, ios::out);
		if(!myfile2)
		{
			cout<<"error!"<<endl;
		}
		else
		{
			myfile2<<planeModel[i].at<double>(1,0)<<" "<<planeModel[i].at<double>(1,1)<<" "<<planeModel[i].at<double>(1,2)<<endl;
			myfile2<<planeModel[i].at<double>(0,0)<<" "<<planeModel[i].at<double>(0,1)<<" "<<planeModel[i].at<double>(0,2)<<endl;
		}
		myfile2.close();
	}

	//write all the horizontal planes into one ply file with different color
	/*int colorValue[3]={50, 150, 255};
	int RGBValue[27][3];
	int rgbn=0;
	for (int i=0; i<3; i++)
	{
		for (int j=0; j<3; j++)
		{
			for (int k=0; k<3; k++)
			{
				RGBValue[rgbn][0]=colorValue[i];
				RGBValue[rgbn][1]=colorValue[j];
				RGBValue[rgbn][2]=colorValue[k];
				rgbn++;
			}
		}
	}*/

	//read color from "rgb.txt" which is generated by jet in matlab
	int RGBValue[200][3];
	int rgbn=0;
    char inPath[100]="/home/victor/project_ws/src/ground_plane/src/rgb.txt";
	ifstream infile(inPath, ios::in);
	if(!infile)
	{
		cout<<"error reading rgb.txt!"<<endl;
	}
	else
	{
		for(int i=0; i<200; i++)
		{
			infile>>RGBValue[i][0]>>RGBValue[i][1]>>RGBValue[i][2];
		}
		infile.close();
	}

	//write the horizontal plane into ply file
	char outPath[100];
	sprintf(outPath, "/home/victor/project_ws/src/ground_plane/src/%s/ransacPlane(num:%d).ply", outputFile, int(inliers.size()));
	ofstream myfile0(outPath, ios::out);
	if(!myfile0)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		myfile0<<"ply"<<endl;
		myfile0<<"format ascii 1.0"<<endl;
		myfile0<<"element vertex "<<num_allpoints<<endl;
		myfile0<<"property double x"<<endl;
		myfile0<<"property double y"<<endl;
		myfile0<<"property double z"<<endl;
		myfile0<<"property uchar red"<<endl;
		myfile0<<"property uchar green"<<endl;
		myfile0<<"property uchar blue"<<endl;
		myfile0<<"element face 0"<<endl;
		myfile0<<"end_header"<<endl;
		for (int i=0; i<inliers.size(); i++)
		{
			for(int j=0; j<inliers[i].size(); j++)
			{
				myfile0<<cloud->points[inliers[i][j]].x<<" "<<cloud->points[inliers[i][j]].y<<" "<<cloud->points[inliers[i][j]].z<<" "<<RGBValue[i][0]<<" "<<RGBValue[i][1]<<" "<<RGBValue[i][2]<<endl;
			}
		}
		myfile0.close();

	}
	
	//write the running time into txt file
	char outPath2[100];
	sprintf(outPath2, "/home/victor/project_ws/src/ground_plane/src/%s/time.txt", outputFile);
	ofstream myfile3(outPath2, ios::out);
	if(!myfile3)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		myfile3<<(double)(t[9]-t[2])/CLOCKS_PER_SEC<<" "<<"s"<<endl;
		myfile3.close();

	}

	t[11]=clock();
	//printf("%lf s\n",(double)(t[11]-t[10])/CLOCKS_PER_SEC);
	printf("Find Horizontal Planes Time: %lf s\n", (double)(t[9]-t[2])/CLOCKS_PER_SEC);
	printf("Total Time: %lf s\n",(double)(t[11]-t[1])/CLOCKS_PER_SEC);

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
	//return 0;
}
//get all the files in path
void getFiles(char* dir_name, vector<string> &files)
{
    // check the parameter  
    if(dir_name == NULL)  
    {  
        cout<<"dir_name is null !"<<endl;  
        return;  
    }  
  
    // check if dir_name is a valid dir  
    struct stat s;  
    lstat(dir_name, &s);  
    if(!S_ISDIR(s.st_mode))  
    {  
        cout<<"dir_name is not a valid directory!"<<endl;  
        return;  
    }  
      
    struct dirent* filename;    // return value for readdir()  
    DIR* dir;                   // return value for opendir()  
    dir = opendir(dir_name);  
    if(dir == NULL)  
    {  
        cout<<"Can not open dir "<<dir_name<<endl;  
        return;  
    }  
    cout<<"Successfully opened the dir!"<<endl;  
      
    //read all the files in the dir  
    while((filename = readdir(dir)) != NULL)  
    {  
        // get rid of "." and ".."  
        if( strcmp(filename->d_name, ".") == 0 ||   
            strcmp(filename->d_name, "..") == 0 )  
            continue;  
        //cout<<filename->d_name<<endl;  
        files.push_back(filename->d_name);
    }  
}
int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));
    ros::init(argc, argv, "read_points");
	ros::NodeHandle n;
    string folderName="home_data_ascii";
    //string folderName="office_data_ascii";
	//string folderName="rgbd-scenes-v2";
    char mkFile[100];
    sprintf(mkFile, "/home/victor/project_ws/src/ground_plane/src/%s/", folderName.c_str());
    cout<<mkFile<<endl;
    int isCreate=mkdir(mkFile, S_IRWXU);
    if(!isCreate)
    {
        printf("Create Path:%s\n",mkFile);
    }
    else
    {
        printf("Create Path Failed!\n");
        return 0;
    }
    char filePath[100];
    sprintf(filePath, "/home/victor/project_ws/src/ground_plane/data/%s", folderName.c_str());
    vector<string> files;
    getFiles(filePath, files);
    for(int i=0; i<files.size(); i++)
    {
        cout<<files[i].c_str()<<endl;
        char inPath[100];
        sprintf(inPath, "/home/victor/project_ws/src/ground_plane/data/%s/%s", folderName.c_str(), files[i].c_str());
        char outPath[100];
        string fname;
        fname.assign(files[i].begin(), files[i].end()-4);
        sprintf(outPath, "%s/%s", folderName.c_str(), fname.c_str());
        FindHorizontalPlanes(inPath, outPath);
    }
    return 0;
}
