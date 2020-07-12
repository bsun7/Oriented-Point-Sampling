#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
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


struct plInter
{
	int planeIndex;
	int overlapNum;  //the number of overlap points
	float dis1;  //the distance from centroid of plane 1 to plane 2 
	float dis2;  //the distance from centroid of plane 2 to plane 1
	float interAngle;
};

//insertion sort
void InsertSort(std::vector<plInter> &open, int low, int high)  
{  
    for(int i=(low+1); i<=high; i++)
    {  
        if(open[i].overlapNum < open[i-1].overlapNum)
        {               
			int j=i-1;
			struct plInter temp=open[i];   
            while(j>=low && temp.overlapNum<open[j].overlapNum)
            {   
				open[j+1]=open[j];
                j--;         
			}  
			open[j+1]=temp;  
        }            
    }    
}  

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

//compute sigma and NWR matrix, then get the normals of the points
pcl::PointXYZ NormalEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointXYZ p0, int maxK, std::vector<int> pointIdxRadiusSearch, std::vector<float> pointRadiusSquaredDistance)
{
	cv::Mat diffM(3, 1, CV_64FC1);
	cv::Mat referP(3, 1, CV_64FC1);
	cv::Mat neighP(3, 1, CV_64FC1);
	cv::Mat nwrSample(3, 3, CV_64FC1);
	cv::Mat nwrMat(3, 3, CV_64FC1);
	pcl::PointXYZ normal;
	float sigma=0.2;
	//compute NWR Matrix
	referP.at<float>(0,0)=p0.x;
	referP.at<float>(1,0)=p0.y;
	referP.at<float>(2,0)=p0.z;
	nwrMat=cv::Mat::zeros(cv::Size(3,3),CV_64FC1);
	for(int i=1; i<maxK; i++)
	{
		neighP.at<float>(0,0)=cloud->points[pointIdxRadiusSearch[i]].x;
		neighP.at<float>(1,0)=cloud->points[pointIdxRadiusSearch[i]].y;
		neighP.at<float>(2,0)=cloud->points[pointIdxRadiusSearch[i]].z;
		diffM=neighP-referP;
		nwrSample=diffM*diffM.t();
		nwrSample=nwrSample/(pointRadiusSquaredDistance[i]*pointRadiusSquaredDistance[i]);
		nwrSample=nwrSample*exp(-(pointRadiusSquaredDistance[i]*pointRadiusSquaredDistance[i])/(2*sigma*sigma));
		nwrMat=nwrMat+nwrSample;	
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
	normal.x=eigenvector.at<float>(2,0);
	normal.y=eigenvector.at<float>(2,1);
	normal.z=eigenvector.at<float>(2,2);
	return normal;
}

void FSPFHorizontalPlanes(char* inputFile, char* outputFile)
{
	//srand((unsigned)time(NULL));
	//ros::init(argc, argv, "FSPF");
	//ros::NodeHandle n;
	t[1]=clock();
	//read 3D points from pcd file
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	//char fileName[100]="/home/victor/project_ws/src/ground_plane/data/home_data_ascii/scene1_ascii.pcd";
	cloud = ReadPCD(inputFile);
	int pointNum=cloud->points.size();  //number of points in the cloud
	cout<<"Points number in the cloud: "<<pointNum<<endl;
	//write all the points in cloud to ply file
	char mkFile[100];
    sprintf(mkFile, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s", outputFile);
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
    sprintf(outFile1, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/allPoints.ply", outputFile);
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
		myfile1<<"property float x"<<endl;
		myfile1<<"property float y"<<endl;
		myfile1<<"property float z"<<endl;
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
	t[2]=clock();
	printf("read pcd and write allPoints.ply: %lf s\n",(double)(t[2]-t[1])/CLOCKS_PER_SEC);
	//Fast Sampling Plane Filtering
	std::vector<std::vector<pcl::PointXYZ> > Planes;  //store the points of the planes FSPF find
	std::vector<std::vector<int> > inliers; //store the index of points in each plane
	std::vector<pcl::PointXYZ> Normals;  //store the normals of the planes FSPF find(re-estimated)
	//std::vector<pcl::PointXYZ> Normals_f;  //store the normals of the planes FSPF find
	std::vector<pcl::PointXYZ> Centers;  //store the centers of the planes FSPF find
	std::vector<std::vector<pcl::PointXYZ> > Outliers;  // store the outliers points
	//Parameters of FSPF
	int numPoints=0;  //number of the points on the planes
	int kSample=-1;  //number of neighbor samples
	int nMax=2*pointNum;  //maximum of the points number on the planes
	int kMax=pointNum;  //maximum of the neighbor samples number
	int numLocal=40;  //number of local samples
	float errThresh=0.004;  //plane offset error for inliers
	float minRate=0.8;  //minimum fraction of inliers to accept local sample
	float radius1=0.01;  //radius within to find neighbor p1 and p2
	float radius2=0.02;  //radius within to find inliers
	//construct the kd-tree using the point cloud
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud (cloud);

	std::vector<int> randN;
	for (int i=0; i<pointNum; i++)
	{
		randN.push_back(i);
	}
	random_shuffle(randN.begin(), randN.end());
	//declare a vector to record if a point is used
	/*std::vector<bool> isUsed;
	for (int i=0; i<pointNum; i++)
	{
		isUsed.push_back(false);
	}*/
	while (numPoints<nMax && kSample<kMax)
	{
		t[3]=clock();
		kSample++;
		/*while(isUsed[randN[kSample]]==true)
		{
			kSample++;
		}*/
		cout<<endl<<"kSample= "<<kSample<<endl;
		pcl::PointXYZ p0, p1, p2;
		//int randInd0=rand()%pointNum;
		int randInd0=randN[kSample];
		cout<<"randInd0= "<<randInd0<<endl;
		p0=cloud->points[randInd0];
		//use radius search of kdtree
		std::vector<int> pointIdxRadiusSearch;
    	std::vector<float> pointRadiusSquaredDistance;
		if(kdtree.radiusSearch(p0, radius1, pointIdxRadiusSearch, pointRadiusSquaredDistance)>0)
		{
			cout<<"Neighbors within radius("<<radius1<<") search finds "<<pointIdxRadiusSearch.size()<<" points"<<endl;
		}
		/*//use k nearest neighbors of kdtree
		std::vector<int> pointIdxKNNSearch;
    	std::vector<float> pointKNNSquaredDistance;
		if(kdtree.nearestKSearch (p0, 41, pointIdxKNNSearch, pointKNNSquaredDistance)>0)
		{
			cout<<"Neighbors within KNN("<<40<<") search finds "<<pointIdxKNNSearch.size()<<" points"<<endl;
		}*/

		/*//cout pointIdxKNNSearch and pointKNNSquaredDistance
		cout<<"pointIdxKNNSearch:"<<endl;
		for (int i=0; i<pointIdxKNNSearch.size(); i++)
		{
			cout<<pointIdxKNNSearch[i]<<" ";
		}
		cout<<endl;

		cout<<"pointKNNSquaredDistance:"<<endl;
		for (int i=0; i<pointKNNSquaredDistance.size(); i++)
		{
			cout<<pointKNNSquaredDistance[i]<<" ";
		}
		cout<<endl;*/

		t[4]=clock();
		printf("%lf s\n",(double)(t[4]-t[3])/CLOCKS_PER_SEC);

		pcl::PointXYZ r; //the normal of the plane
		//use more neighbors to compute the normal
		/*if (pointIdxRadiusSearch.size()>=20)
		{
			r=NormalEstimation(cloud, p0, 20, pointIdxRadiusSearch, pointRadiusSquaredDistance);
		}
		else
		{
			r=NormalEstimation(cloud, p0, pointIdxRadiusSearch.size(), pointIdxRadiusSearch, pointRadiusSquaredDistance);
		}*/

		//r=NormalEstimation(cloud, p0, 41, pointIdxKNNSearch, pointKNNSquaredDistance);
		//cout<<"r= "<<r.x<<" "<<r.y<<" "<<r.z<<endl;

		//use cross product to compute the normal
		int randInd1=rand()%pointIdxRadiusSearch.size();
		int randInd2=rand()%pointIdxRadiusSearch.size();
		p1=cloud->points[pointIdxRadiusSearch[randInd1]];
		p2=cloud->points[pointIdxRadiusSearch[randInd2]];
		float p10x=p1.x-p0.x;
		float p10y=p1.y-p0.y;
		float p10z=p1.z-p0.z;
		float p20x=p2.x-p0.x;
		float p20y=p2.y-p0.y;
		float p20z=p2.z-p0.z;
		r.x=p10y*p20z-p10z*p20y;
		r.y=p10z*p20x-p10x*p20z;
		r.z=p10x*p20y-p10y*p20x;
		float rn=sqrt(r.x*r.x+r.y*r.y+r.z*r.z);
		r.x=r.x/rn;
		r.y=r.y/rn;
		r.z=r.z/rn;

		int numInliers=0;
		std::vector<int> pointIdx;
    	std::vector<float> pointDistance;
		t[5]=clock();
		printf("%lf s\n",(double)(t[5]-t[4])/CLOCKS_PER_SEC);
		if(kdtree.radiusSearch(p0, radius2, pointIdx, pointDistance)>0)
		{
			cout<<"Neighbors within radius("<<radius2<<") search finds "<<pointIdx.size()<<" points"<<endl;
		}
		t[6]=clock();
		printf("%lf s\n",(double)(t[6]-t[5])/CLOCKS_PER_SEC);
		if (pointIdx.size()<numLocal)
		{
			continue;
		}
		std::random_shuffle(pointIdx.begin(), pointIdx.end());
		int num=0;
		std::vector<pcl::PointXYZ> pl;
		pcl::PointXYZ nm;
		pcl::PointXYZ center;
		center.x=0;
		center.y=0;
		center.z=0;
		std::vector<int> inl;
		inl.push_back(randInd0);
		inl.push_back(pointIdxRadiusSearch[randInd1]);
		inl.push_back(pointIdxRadiusSearch[randInd2]);
		pl.push_back(p0);
		pl.push_back(p1);
		pl.push_back(p2);
		numInliers+=3;
		for (int i=0; i<numLocal; i++) 
		{
			if ((pointIdx[i]==randInd0) || (pointIdx[i]==pointIdxRadiusSearch[randInd1]) || (pointIdx[i]==pointIdxRadiusSearch[randInd2]))
			{
				continue;
			}
			num++;
			/*if (isUsed[pointIdx[i]]==true)
			{
				continue;
			}*/
			pcl::PointXYZ pi;
			pi=cloud->points[pointIdx[i]];
			float pi0x=pi.x-p0.x;
			float pi0y=pi.y-p0.y;
			float pi0z=pi.z-p0.z;
			float e=abs(r.x*pi0x+r.y*pi0y+r.z*pi0z);
			if (e<errThresh)
			{
				pl.push_back(pi);
				inl.push_back(pointIdx[i]);
				numInliers++;
			}
			if (num==numLocal-3)
			{
				break;
			}
		}

		if (numInliers>(minRate*numLocal))
		{
			Planes.push_back(pl);
			inliers.push_back(inl);
			
			/*for (int i=0; i<inliers.size(); i++)
			{
				isUsed[inliers[i]]=true;
			}*/

			//compute the center point of all the inliers points
			for (int i=0; i<pl.size(); i++)
			{
				center.x+=pl[i].x;
				center.y+=pl[i].y;
				center.z+=pl[i].z;
			}
			center.x=center.x/pl.size();
			center.y=center.y/pl.size();
			center.z=center.z/pl.size();
			Centers.push_back(center);

			//re-estimate the plane normal using all the inliers points
			cv::Mat pointsInlier(3, pl.size(), CV_32FC1);
			for(int i=0; i<pl.size(); i++)
			{
				pointsInlier.at<float>(0,i)=pl[i].x-center.x;
				pointsInlier.at<float>(1,i)=pl[i].y-center.y;
				pointsInlier.at<float>(2,i)=pl[i].z-center.z;
			}
			cv::Mat u,w,v;
			SVD::compute(pointsInlier, w, u, v);
			nm.x=u.at<float>(0,2);
			nm.y=u.at<float>(1,2);
			nm.z=u.at<float>(2,2);

			Normals.push_back(nm);

			//normals from three point cross product
			//Normals.push_back(r);

			numPoints+=numInliers;
		}
		else
		{
			Outliers.push_back(pl);
		}
		t[7]=clock();
		printf("%lf s\n",(double)(t[7]-t[6])/CLOCKS_PER_SEC);
	}
	cout<<endl<<"Planes number: "<<Planes.size()<<endl;
	cout<<"Planes filtered points number: "<<numPoints<<endl;
	t[8]=clock();
	printf("FSPF time: %lf s\n",(double)(t[8]-t[2])/CLOCKS_PER_SEC);

	/*//write all the planes into ply file
	char outPath1[100];
	sprintf(outPath1, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/FSPFPlane(num:%d).ply", outputFile, int(Planes.size()));
	ofstream myfile2(outPath1, ios::out);
	if(!myfile2)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		myfile2<<"ply"<<endl;
		myfile2<<"format ascii 1.0"<<endl;
		myfile2<<"element vertex "<<numPoints<<endl;
		myfile2<<"property float x"<<endl;
		myfile2<<"property float y"<<endl;
		myfile2<<"property float z"<<endl;
		myfile2<<"property uchar red"<<endl;
		myfile2<<"property uchar green"<<endl;
		myfile2<<"property uchar blue"<<endl;
		myfile2<<"element face 0"<<endl;
		myfile2<<"end_header"<<endl;
		for (int i=0; i<Planes.size(); i++)
		{
			for(int j=0; j<Planes[i].size(); j++)
			{
				myfile2<<Planes[i][j].x<<" "<<Planes[i][j].y<<" "<<Planes[i][j].z<<" "<<"255 0 0"<<endl;
			}
		}
		myfile2.close();
	}

	//write the normals into txt file
	char outFile3[100];
    sprintf(outFile3, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/normals.txt", outputFile);
	ofstream myfile6(outFile3, ios::out);
	if(!myfile6)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		for(int i=0; i<Normals.size(); i++)
		{
			myfile6<<Normals[i].x<<" "<<Normals[i].y<<" "<<Normals[i].z<<endl;
		}
		myfile6.close();
	}*/

	//parameters for horizontal planes
	float interAngle_thre1=5;
	float interAngle_thre2=10;

	//find the horizontal planes
	float ab, an, bn, cosr, interAngle;
	std::vector<std::vector<pcl::PointXYZ> > horiPlanes;  //store horizontal planes
	int curPlaneId_hori=-1;  //the index of current horizontal plane
	std::vector<std::vector<int> > planeList;  //store the list of horizontal planes which each point belongs to(all the points) 
	for (int i=0; i<pointNum; i++)
	{
		std::vector<int> plist;
		plist.push_back(-1);
		planeList.push_back(plist);
	}
	std::vector<std::vector<int> > inliers_hori;  //store the index of inliers on the horizontal plane
	std::vector<pcl::PointXYZ> horiNormals;  //store the normals of the planes FSPF find(re-estimated)
	//std::vector<pcl::PointXYZ> horiNormals_f;  ////store the normals of the planes FSPF find
	std::vector<pcl::PointXYZ> horiCenters;
	int num_hori=0;  //store the number of all points on horizontal planes
	for (int i=0; i<Planes.size(); i++)
	{
		//compute the angle between normal and yAxis direction (0,1,0)/zAxis direction (0,0,1)
		ab=Normals[i].z;
		an=1;
		bn=sqrt(Normals[i].x*Normals[i].x+Normals[i].y*Normals[i].y+Normals[i].z*Normals[i].z);
		cosr=ab/(an*bn);
		interAngle=acos(cosr)*180/M_PI;
		if(interAngle<interAngle_thre1 || interAngle>(180-interAngle_thre1))
		{
			horiPlanes.push_back(Planes[i]);
			horiNormals.push_back(Normals[i]);
			//horiNormals_f.push_back(Normals_f[i]);
			num_hori+=Planes[i].size();
			curPlaneId_hori++;
			inliers_hori.push_back(inliers[i]);
			for (int j=0; j<inliers[i].size(); j++)
			{
				if (planeList[inliers[i][j]][0]==-1)
				{
					planeList[inliers[i][j]][0]=curPlaneId_hori;
				}
				else
				{
					planeList[inliers[i][j]].push_back(curPlaneId_hori);
				}
			}
			//add center points
			horiCenters.push_back(Centers[i]);
		}
	}
	cout<<"Horizontal Planes number: "<<horiPlanes.size()<<endl;
	t[9]=clock();


	/*//write the normals into txt file
	char normPath[100];
    sprintf(normPath, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/normals.txt", outputFile);
	ofstream normfile(normPath, ios::out);
	if(!normfile)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		for(int i=0; i<horiNormals.size(); i++)
		{
			normfile<<horiNormals[i].x<<" "<<horiNormals[i].y<<" "<<horiNormals[i].z<<endl;
		}
		normfile.close();
	}

	char normPath1[100];
    sprintf(normPath1, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/normals_f.txt", outputFile);
	ofstream normfile1(normPath1, ios::out);
	if(!normfile1)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		for(int i=0; i<horiNormals_f.size(); i++)
		{
			normfile1<<horiNormals_f[i].x<<" "<<horiNormals_f[i].y<<" "<<horiNormals_f[i].z<<endl;
		}
		normfile1.close();
	}*/

	//only keep the points on the horizontal planes in planeList
	std::vector<std::vector<int> > planeList_hori;  //store the list of horizontal planes which each point belongs to(only horizontal points) 
	std::vector<int> index_hori;  //store the index of the points in planeList_hori
	for (int i=0; i<planeList.size(); i++)
	{
		if (planeList[i][0]==-1)
		{
			continue;
		}
		planeList_hori.push_back(planeList[i]);
		index_hori.push_back(i);
	}

	//compute the plane intersection matrix 
	std::vector<std::vector<int> > planeInterMatrix;
	for (int i=0; i<horiPlanes.size(); i++)
	{
		std::vector<int> plim;
		for (int j=0; j<horiPlanes.size(); j++)
		{
			plim.push_back(0);
		}
		planeInterMatrix.push_back(plim);
	}

	for (int i=0; i<planeList_hori.size(); i++)
	{
		for (int j=0; j<planeList_hori[i].size()-1; j++)
		{
			for (int k=j+1; k<planeList_hori[i].size(); k++)
			{
				planeInterMatrix[planeList_hori[i][j]][planeList_hori[i][k]]++;
				planeInterMatrix[planeList_hori[i][k]][planeList_hori[i][j]]++;
			}
		}
	}
	
	//record the centroid-to-plane distance and the angle between normals(for non-zeros in planeInterMatrix)

	std::vector<std::vector<plInter> > planeInterDisAng;  //record the distance and angle between normals
	for (int i=0; i<planeInterMatrix.size(); i++)
	{
		std::vector<plInter> plida;
		for (int j=0; j<planeInterMatrix[i].size(); j++)
		{
			if (planeInterMatrix[i][j]!=0 && i<j)
			{
				plInter pli;
				pli.planeIndex=j;
				pli.overlapNum=planeInterMatrix[i][j];
				//compute dis1
				float x1=horiCenters[i].x-horiPlanes[j][0].x;
				float y1=horiCenters[i].y-horiPlanes[j][0].y;
				float z1=horiCenters[i].z-horiPlanes[j][0].z;
				float xn1=horiNormals[j].x;
				float yn1=horiNormals[j].y;
				float zn1=horiNormals[j].z;
				float norm_n1=sqrt(xn1*xn1+yn1*yn1+zn1*zn1);
				pli.dis1=abs(x1*xn1+y1*yn1+z1*zn1)/norm_n1;
				//compute dis2
				float x2=horiCenters[j].x-horiPlanes[i][0].x;
				float y2=horiCenters[j].y-horiPlanes[i][0].y;
				float z2=horiCenters[j].z-horiPlanes[i][0].z;
				float xn2=horiNormals[i].x;
				float yn2=horiNormals[i].y;
				float zn2=horiNormals[i].z;
				float norm_n2=sqrt(xn2*xn2+yn2*yn2+zn2*zn2);
				pli.dis2=abs(x2*xn2+y2*yn2+z2*zn2)/norm_n2;
				//compute intersection angle
				float n1n2=horiNormals[i].x*horiNormals[j].x+horiNormals[i].y*horiNormals[j].y+horiNormals[i].z*horiNormals[j].z;
				float n1=sqrt(horiNormals[i].x*horiNormals[i].x+horiNormals[i].y*horiNormals[i].y+horiNormals[i].z*horiNormals[i].z);
				float n2=sqrt(horiNormals[j].x*horiNormals[j].x+horiNormals[j].y*horiNormals[j].y+horiNormals[j].z*horiNormals[j].z);
				float cosnn=n1n2/(n1*n2);
				//for the number which is a little bigger than interAngle or a little smaller than interAngle
				if (cosnn>1)
				{
					pli.interAngle=0;
				}
				else
				{
					if (cosnn<-1)
					{
						pli.interAngle=180;
					}
					else
					{
						pli.interAngle=acos(cosnn)*180/M_PI;
					}
				}
				plida.push_back(pli);
			}
		}
		if (!plida.empty())
		{
			planeInterDisAng.push_back(plida);
		}
		else
		{
			plInter pli;
			pli.planeIndex=-1;
			pli.dis1=-1;
			pli.dis2=-1;
			pli.interAngle=-1;
			plida.push_back(pli);
			planeInterDisAng.push_back(plida);
		}
	}

	//write planeInterDisAng into txt file
	char outPath7[100];
	sprintf(outPath7, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/planeInterDisAng.txt", outputFile);
	ofstream myfile9(outPath7, ios::out);
	if(!myfile9)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		for (int i=0; i<planeInterDisAng.size(); i++)
		{
			if (planeInterDisAng[i][0].planeIndex==-1)
			{
				myfile9<<"plane "<<i<<": none"<<endl;
			}
			else
			{
				myfile9<<"plane "<<i<<": "<<endl;
				for (int j=0; j<planeInterDisAng[i].size(); j++)
				{
					myfile9<<"         plane "<<planeInterDisAng[i][j].planeIndex<<" overlapNum: "<<planeInterDisAng[i][j].overlapNum<<" dis1: "<<planeInterDisAng[i][j].dis1
					                          <<" dis2: "<<planeInterDisAng[i][j].dis2<<" intersection angle: "<<planeInterDisAng[i][j].interAngle<<endl;
				}
			}
			myfile9<<endl;
		}
		myfile9.close();
	}
	

	/*//write each horizontal plane into ply for testing
	for (int i=0; i<horiPlanes.size(); i++)
	{
		int tnum=horiPlanes[i].size();
		char toutPath[100];
		sprintf(toutPath, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/FSPFPlane_hori_%d.ply", outputFile, i);
		ofstream tfile(toutPath, ios::out);
		if(!tfile)
		{
			cout<<"error!"<<endl;
		}
		else
		{
			tfile<<"ply"<<endl;
			tfile<<"format ascii 1.0"<<endl;
			tfile<<"element vertex "<<tnum<<endl;
			tfile<<"property float x"<<endl;
			tfile<<"property float y"<<endl;
			tfile<<"property float z"<<endl;
			tfile<<"property uchar red"<<endl;
			tfile<<"property uchar green"<<endl;
			tfile<<"property uchar blue"<<endl;
			tfile<<"element face 0"<<endl;
			tfile<<"end_header"<<endl;
		
			for(int j=0; j<horiPlanes[i].size(); j++)
			{
				tfile<<horiPlanes[i][j].x<<" "<<horiPlanes[i][j].y<<" "<<horiPlanes[i][j].z<<" "<<"255 0 0"<<endl;
			}
		}
	}*/
	


	int num_merge_s=0;
	int num_merge_f=0;
	//combine the planes with overlap points
	std::vector<bool> isExist;  //record the merging planes
	for (int i=0; i<planeInterDisAng.size(); i++)
	{
		isExist.push_back(true);
	}
	cout<<"Start Merging..."<<endl;
	for (int i=0; i<planeInterDisAng.size(); i++)
	{
		//cout<<"Merge Plane "<<i<<"..."<<endl;
		if (planeInterDisAng[i][0].planeIndex==-1 || isExist[i]==false)
		{
			continue;
		}
		//InsertSort(planeInterDisAng[i], 0, planeInterDisAng[i].size()-1);
		int j=1;  //the number of processed neighbors
		std::vector<int> hasTest;  
		for (int n=0; n<planeInterDisAng[i].size(); n++)
		{
			hasTest.push_back(false);
		}
		while(j<=planeInterDisAng[i].size())
		{
			cout<<"Merge Plane "<<i<<": j = "<<j<<" ";
			//find the neighbor plane with the largest overlap points number
			int pnMax;
			int overlapMax=-1;
			for (int k=0; k<planeInterDisAng[i].size(); k++)
			{
				if (hasTest[k])
				{
					continue;
				}
				if (planeInterDisAng[i][k].overlapNum>overlapMax)
				{
					overlapMax=planeInterDisAng[i][k].overlapNum;
					pnMax=k;
				}
			}
			cout<<"pnMax = "<<pnMax<<endl;
			//coplanar test
			int indMax=planeInterDisAng[i][pnMax].planeIndex;
			if ((planeInterDisAng[i][pnMax].dis1<errThresh || planeInterDisAng[i][pnMax].dis2<errThresh) && (planeInterDisAng[i][pnMax].interAngle<interAngle_thre2 || planeInterDisAng[i][pnMax].interAngle>(180-interAngle_thre2)))
			{
				num_merge_s++;
				//merge planes and update
				//merge the two planes
				std::vector<int> inliers_new;
				sort(inliers_hori[i].begin(), inliers_hori[i].end());
				sort(inliers_hori[indMax].begin(), inliers_hori[indMax].end());
				set_union(inliers_hori[i].begin(), inliers_hori[i].end(), inliers_hori[indMax].begin(), inliers_hori[indMax].end(), back_inserter(inliers_new));
				inliers_hori[i]=inliers_new;
				std::vector<pcl::PointXYZ> hp_new;
				for (int k=0; k<inliers_new.size(); k++)
				{
					hp_new.push_back(cloud->points[inliers_new[k]]);
				}
				horiPlanes[i]=hp_new;
				//delete plane indMax
				isExist[indMax]=false;
				//compute the centroid and normal of the new plane
				pcl::PointXYZ pt;
				pt.x=0;
				pt.y=0;
				pt.z=0;
				for (int k=0; k<hp_new.size(); k++)
				{
					pt.x+=hp_new[k].x;
					pt.y+=hp_new[k].y;
					pt.z+=hp_new[k].z;
				}
				horiCenters[i].x=pt.x/hp_new.size();
				horiCenters[i].y=pt.y/hp_new.size();
				horiCenters[i].z=pt.z/hp_new.size();

				cv::Mat pointsInlier(3, hp_new.size(), CV_32FC1);
				for(int k=0; k<hp_new.size(); k++)
				{
					pointsInlier.at<float>(0,k)=hp_new[k].x-horiCenters[i].x;
					pointsInlier.at<float>(1,k)=hp_new[k].y-horiCenters[i].y;
					pointsInlier.at<float>(2,k)=hp_new[k].z-horiCenters[i].z;
				}
				cv::Mat u,w,v;
				SVD::compute(pointsInlier, w, u, v);
				horiNormals[i].x=u.at<float>(0,2);
				horiNormals[i].y=u.at<float>(1,2);
				horiNormals[i].z=u.at<float>(2,2);

				//update planeInterDisAng 
				//combine planeIndex of row i and row indMax in planeInterDisAng
				planeInterDisAng[i].erase(planeInterDisAng[i].begin()+pnMax);
				std::vector<int> plset1, plset2;
				for (int k=0; k<planeInterDisAng[i].size(); k++)
				{
					plset1.push_back(planeInterDisAng[i][k].planeIndex);
				}
				if (planeInterDisAng[indMax][0].planeIndex!=-1)
				{
					for (int k=0; k<planeInterDisAng[indMax].size(); k++)
					{
						plset2.push_back(planeInterDisAng[indMax][k].planeIndex);
					}
				}
				std::vector<int> plset_new1;
				sort(plset1.begin(), plset1.end());
				/*cout<<"plset1:"<<endl;
				for (int k=0; k<plset1.size(); k++)
				{
					cout<<"       plane "<<plset1[k]<<endl;
				}*/
				sort(plset2.begin(), plset2.end());
				/*cout<<"plset2:"<<endl;
				for (int k=0; k<plset2.size(); k++)
				{
					cout<<"       plane "<<plset2[k]<<endl;
				}*/
				set_union(plset1.begin(), plset1.end(), plset2.begin(), plset2.end(), back_inserter(plset_new1));
				/*cout<<"plset_new:"<<endl;
				for (int k=0; k<plset_new.size(); k++)
				{
					cout<<"       plane "<<plset_new[k]<<endl;
				}*/
				
				//find the row which has neighbor plane indMax and update
				std::vector<int> plset3;
				for (int k=i+1; k<indMax; k++)
				{
					if (isExist[k]==false)
					{
						continue;
					}
					for (int l=0; l<planeInterDisAng[k].size(); l++)
					{
						if (planeInterDisAng[k][l].planeIndex==indMax)
						{
							//romove th/home/victor/project_ws/src/ground_plane/src/FSPF.cpp:315:31:e neighbor plane indMax and store the row index
							planeInterDisAng[k].erase(planeInterDisAng[k].begin()+l);
							plset3.push_back(k);
							/*//update the neighbor plane indMax
							planeInterDisAng[k][l].planeIndex=i;
							sort(inliers_hori[k].begin(), inliers_hori[k].end());
							std::vector<int> inter;
							set_intersection(inliers_hori[k].begin(), inliers_hori[k].end(), inliers_hori[i].begin(), inliers_hori[i].end(), back_inserter(inter));
							planeInterDisAng[k][l].overlapNum=inter.size();
							//compute dis1
							float x1=horiCenters[k].x-horiPlanes[indMax][0].x;
							float y1=horiCenters[k].y-horiPlanes[indMax][0].y;
							float z1=horiCenters[k].z-horiPlanes[indMax][0].z;
							float xn1=horiNormals[indMax].x;
							float yn1=horiNormals[indMax].y;
							float zn1=horiNormals[indMax].z;
							float norm_n1=sqrt(xn1*xn1+yn1*yn1+zn1*zn1);
							planeInterDisAng[k][l].dis1=abs(x1*xn1+y1*yn1+z1*zn1)/norm_n1;
							//compute dis2
							float x2=horiCenters[indMax].x-horiPlanes[k][0].x;
							float y2=horiCenters[indMax].y-horiPlanes[k][0].y;
							float z2=horiCenters[indMax].z-horiPlanes[k][0].z;
							float xn2=horiNormals[k].x;
							float yn2=horiNormals[k].y;
							float zn2=horiNormals[k].z;
							float norm_n2=sqrt(xn2*xn2+yn2*yn2+zn2*zn2);
							planeInterDisAng[k][l].dis2=abs(x2*xn2+y2*yn2+z2*zn2)/norm_n2;
							//compute intersection angle
							float n1n2=horiNormals[k].x*horiNormals[indMax].x+horiNormals[k].y*horiNormals[indMax].y+horiNormals[k].z*horiNormals[indMax].z;
							float n1=sqrt(horiNormals[k].x*horiNormals[k].x+horiNormals[k].y*horiNormals[k].y+horiNormals[k].z*horiNormals[k].z);
							float n2=sqrt(horiNormals[indMax].x*horiNormals[indMax].x+horiNormals[indMax].y*horiNormals[indMax].y+horiNormals[indMax].z*horiNormals[indMax].z);
							float cosnn=n1n2/(n1*n2);
							planeInterDisAng[k][l].interAngle=acos(cosnn)*180/M_PI;*/
						}
					}
				}
				std::vector<int> plset_new;
				set_union(plset_new1.begin(), plset_new1.end(), plset3.begin(), plset3.end(), back_inserter(plset_new));
				//update planeInterDisAng[i]
				if (plset_new.size()==0)
				{
					plInter pli0;
					pli0.planeIndex=-1;
					pli0.overlapNum=-1;
					pli0.dis1=-1;
					pli0.dis2=-1;
					pli0.interAngle=-1;
					planeInterDisAng[i].push_back(pli0);
				}
				else
				{
					std::vector<plInter> plida;
					//compute the information of the new neighbors(combined)
					for (int k=0; k<plset_new.size(); k++)
					{
						plInter pli;
						pli.planeIndex=plset_new[k];
						sort(inliers_hori[plset_new[k]].begin(), inliers_hori[plset_new[k]].end());
						std::vector<int> inter;
						set_intersection(inliers_hori[i].begin(), inliers_hori[i].end(), inliers_hori[plset_new[k]].begin(), inliers_hori[plset_new[k]].end(), back_inserter(inter));
						pli.overlapNum=inter.size();
						//compute dis1
						float x1=horiCenters[i].x-horiPlanes[plset_new[k]][0].x;
						float y1=horiCenters[i].y-horiPlanes[plset_new[k]][0].y;
						float z1=horiCenters[i].z-horiPlanes[plset_new[k]][0].z;
						float xn1=horiNormals[plset_new[k]].x;
						float yn1=horiNormals[plset_new[k]].y;
						float zn1=horiNormals[plset_new[k]].z;
						float norm_n1=sqrt(xn1*xn1+yn1*yn1+zn1*zn1);
						pli.dis1=abs(x1*xn1+y1*yn1+z1*zn1)/norm_n1;
						//compute dis2
						float x2=horiCenters[plset_new[k]].x-horiPlanes[i][0].x;
						float y2=horiCenters[plset_new[k]].y-horiPlanes[i][0].y;
						float z2=horiCenters[plset_new[k]].z-horiPlanes[i][0].z;
						float xn2=horiNormals[i].x;
						float yn2=horiNormals[i].y;
						float zn2=horiNormals[i].z;
						float norm_n2=sqrt(xn2*xn2+yn2*yn2+zn2*zn2);
						pli.dis2=abs(x2*xn2+y2*yn2+z2*zn2)/norm_n2;
						//compute intersection angle
						float n1n2=horiNormals[i].x*horiNormals[plset_new[k]].x+horiNormals[i].y*horiNormals[plset_new[k]].y+horiNormals[i].z*horiNormals[plset_new[k]].z;
						float n1=sqrt(horiNormals[i].x*horiNormals[i].x+horiNormals[i].y*horiNormals[i].y+horiNormals[i].z*horiNormals[i].z);
						float n2=sqrt(horiNormals[plset_new[k]].x*horiNormals[plset_new[k]].x+horiNormals[plset_new[k]].y*horiNormals[plset_new[k]].y+horiNormals[plset_new[k]].z*horiNormals[plset_new[k]].z);
						float cosnn=n1n2/(n1*n2);
						pli.interAngle=acos(cosnn)*180/M_PI;
						plida.push_back(pli);
					}
					planeInterDisAng[i]=plida;
				}
				//reset the hasTest
				if (planeInterDisAng[i][0].planeIndex==-1)
				{
					break;
				}
				hasTest.resize(planeInterDisAng[i].size());
				for (int k=0; k<hasTest.size(); k++)
				{
					hasTest[k]=false;
				}
				//reset j
				j=1;

				//for degug: cout updated planeInterDisAng
				cout<<"plane "<<i<<": "<<endl;
				for (int k=0; k<planeInterDisAng[i].size(); k++)
				{
					cout<<"         plane "<<planeInterDisAng[i][k].planeIndex<<" overlapNum: "<<planeInterDisAng[i][k].overlapNum<<" dis1: "<<planeInterDisAng[i][k].dis1
					                          <<" dis2: "<<planeInterDisAng[i][k].dis2<<" intersection angle: "<<planeInterDisAng[i][k].interAngle<<endl;
				}
				
			}
			else
			{
				num_merge_f++;
				hasTest[pnMax]=true;
				j++;
			}
		}
	}

	cout<<"Successfully merging number: "<<num_merge_s<<endl;
	cout<<"Unsuccessfully merging number: "<<num_merge_f<<endl;
	//write plane lists into txt file
	char outPath4[100];
	sprintf(outPath4, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/planeList_hori.txt", outputFile);
	ofstream myfile7(outPath4, ios::out);
	if(!myfile7)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		for (int i=0; i<planeList_hori.size(); i++)
		{
			for (int j=0; j<planeList_hori[i].size(); j++)
			{
				myfile7<<planeList_hori[i][j]<<" ";
			}
			myfile7<<endl;
		}
		myfile7.close();
	}

	//write planeInterMatrix into txt file
	char outPath6[100];
	sprintf(outPath6, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/planeInterMatrix.txt", outputFile);
	ofstream myfile8(outPath6, ios::out);
	if(!myfile8)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		for (int i=0; i<planeInterMatrix.size(); i++)
		{
			for (int j=0; j<planeInterMatrix[i].size(); j++)
			{
				myfile8<<planeInterMatrix[i][j]<<" ";
			}
			myfile8<<endl;
		}
		myfile8.close();
	}

	/*//write planeInterDisAng into txt file
	char outPath7[100];
	sprintf(outPath7, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/planeInterDisAng.txt", outputFile);
	ofstream myfile9(outPath7, ios::out);
	if(!myfile9)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		for (int i=0; i<planeInterDisAng.size(); i++)
		{
			if (planeInterDisAng[i][0].planeIndex==-1)
			{
				myfile9<<"plane "<<i<<": none"<<endl;
			}
			else
			{
				myfile9<<"plane "<<i<<": "<<endl;
				for (int j=0; j<planeInterDisAng[i].size(); j++)
				{
					myfile9<<"         plane "<<planeInterDisAng[i][j].planeIndex<<" overlapNum: "<<planeInterDisAng[i][j].overlapNum<<" dis1: "<<planeInterDisAng[i][j].dis1
					                          <<" dis2: "<<planeInterDisAng[i][j].dis2<<" intersection angle: "<<planeInterDisAng[i][j].interAngle<<endl;
				}
			}
			myfile9<<endl;
		}
		myfile9.close();
	}*/

	int horiNum_merge=accumulate(isExist.begin(), isExist.end(), 0);
	int pointsNum_merge=0;
	for (int i=0; i<horiPlanes.size(); i++)
	{
		if (isExist[i])
		{
			pointsNum_merge+=horiPlanes[i].size();
		}
	}
	//write all the horizontal planes(after merge) into ply file
	char outPath8[100];
	sprintf(outPath8, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/FSPFPlane_hori_merge(num:%d).ply", outputFile, horiNum_merge);
	ofstream myfile10(outPath8, ios::out);
	if(!myfile10)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		myfile10<<"ply"<<endl;
		myfile10<<"format ascii 1.0"<<endl;
		myfile10<<"element vertex "<<pointsNum_merge<<endl;
		myfile10<<"property float x"<<endl;
		myfile10<<"property float y"<<endl;
		myfile10<<"property float z"<<endl;
		myfile10<<"property uchar red"<<endl;
		myfile10<<"property uchar green"<<endl;
		myfile10<<"property uchar blue"<<endl;
		myfile10<<"element face 0"<<endl;
		myfile10<<"end_header"<<endl;
		for (int i=0; i<horiPlanes.size(); i++)
		{
			if (isExist[i])
			{
				for(int j=0; j<horiPlanes[i].size(); j++)
				{
					myfile10<<horiPlanes[i][j].x<<" "<<horiPlanes[i][j].y<<" "<<horiPlanes[i][j].z<<" "<<"255 0 0"<<endl;
				}
			}
		}
		myfile10.close();
	}
	cout<<"Horizontal Planes number: "<<horiPlanes.size()<<endl;
	printf("Horizontal Planes Number(after merge): %d\n", horiNum_merge);

	//write the running time into txt file
	char outPath3[100];
	sprintf(outPath3, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/time.txt", outputFile);
	ofstream myfile4(outPath3, ios::out);
	if(!myfile4)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		myfile4<<(double)(t[9]-t[2])/CLOCKS_PER_SEC<<" "<<"s"<<endl;
		myfile4.close();

	}
	printf("Find Horizontal Planes Time: %lf s\n",(double)(t[9]-t[2])/CLOCKS_PER_SEC);

	//write all the horizontal planes into ply file
	char outPath5[100];
	sprintf(outPath5, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/FSPFPlane_hori(num:%d).ply", outputFile, int(horiPlanes.size()));
	ofstream myfile5(outPath5, ios::out);
	if(!myfile5)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		myfile5<<"ply"<<endl;
		myfile5<<"format ascii 1.0"<<endl;
		myfile5<<"element vertex "<<num_hori<<endl;
		myfile5<<"property float x"<<endl;
		myfile5<<"property float y"<<endl;
		myfile5<<"property float z"<<endl;
		myfile5<<"property uchar red"<<endl;
		myfile5<<"property uchar green"<<endl;
		myfile5<<"property uchar blue"<<endl;
		myfile5<<"element face 0"<<endl;
		myfile5<<"end_header"<<endl;
		for (int i=0; i<horiPlanes.size(); i++)
		{
			for(int j=0; j<horiPlanes[i].size(); j++)
			{
				myfile5<<horiPlanes[i][j].x<<" "<<horiPlanes[i][j].y<<" "<<horiPlanes[i][j].z<<" "<<"255 0 0"<<endl;
			}
		}
		myfile5.close();
	}

	//keep the merged planes and drop the redundant planes
	std::vector<std::vector<pcl::PointXYZ> > horiPlanes_m;
	std::vector<pcl::PointXYZ> horiNormals_m;
	std::vector<pcl::PointXYZ> horiCenters_m;
	for (int i=0; i<horiPlanes.size(); i++)
	{
		if (isExist[i])
		{
			horiPlanes_m.push_back(horiPlanes[i]);
			horiNormals_m.push_back(horiNormals[i]);
			horiCenters_m.push_back(horiCenters[i]);
		}
	}

	//assign all the points in the point cloud to the nearest plane(distance less than threshold) and get final inliers
	std::vector<std::vector<pcl::PointXYZ> > horiPlanes_fi;
	std::vector<pcl::PointXYZ> horiNormals_fi;
	std::vector<pcl::PointXYZ> horiCenters_fi;
	std::vector<std::vector<int> > horiInd;  //store the point index in each plane
	num_hori=0;
	//cout<<"aaaa"<<endl;
	for (int i=0; i<horiPlanes_m.size(); i++)
	{
		std::vector<int> ind;
		ind.push_back(-1);
		horiInd.push_back(ind);
	}
	//cout<<"bbbb"<<endl;
	for(int i=0; i<cloud->points.size(); i++)
	{
		//cout<<"i= "<<i<<endl;
		std::vector<float> dis_in(horiPlanes_m.size(), 0);
		float min_dis;
		int min_ind;
		for(int j=0; j<horiPlanes_m.size(); j++)
		{
			//compute the distance between every point to the inliers_fi planes
			//float x=cloud->points[i].x-planeModel_fi[j].at<float>(0,0);
			//float y=cloud->points[i].y-planeModel_fi[j].at<float>(0,1);
			//float z=cloud->points[i].z-planeModel_fi[j].at<float>(0,2);
			//float xn=planeModel_fi[j].at<float>(1,0);
			//float yn=planeModel_fi[j].at<float>(1,1);
			//float zn=planeModel_fi[j].at<float>(1,2);
			//float norm_n=sqrt(xn*xn+yn*yn+zn*zn);
			//dis_in[j]=abs(x*xn+y*yn+z*zn)/norm_n;
			dis_in[j]=abs(cloud->points[i].z-horiCenters_m[j].z);
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
		if (min_dis<0.01)
		{
			if (horiInd[min_ind][0]==-1)
			{
				horiInd[min_ind][0]=i;
			}
			else
			{
				horiInd[min_ind].push_back(i);
			}
			
		}
	}
	//cout<<"cccc"<<endl;
	for (int i=0; i<horiInd.size(); i++)
	{
		if (horiInd[i][0]!=-1)
		{
			num_hori+=horiInd[i].size();
			//re-compute the center points in inliers_fi
			pcl::PointXYZ cp;
			cp.x=0;
			cp.y=0;
			cp.z=0;
			for (int j=0; j<horiInd[i].size(); j++)
			{
				cp.x+=cloud->points[horiInd[i][j]].x;
				cp.y+=cloud->points[horiInd[i][j]].y;
				cp.z+=cloud->points[horiInd[i][j]].z;
			}
			cp.x=cp.x/horiInd[i].size();
			cp.y=cp.y/horiInd[i].size();
			cp.z=cp.z/horiInd[i].size();
			horiCenters_fi.push_back(cp);

			pcl::PointXYZ hn;
			std::vector<pcl::PointXYZ> hp;
			cv::Mat pointsInlier(3, horiInd[i].size(), CV_32FC1);
			for (int j=0; j<horiInd[i].size(); j++)
			{
				hp.push_back(cloud->points[horiInd[i][j]]);
				//re-compute the normals in inliers_fi
				pointsInlier.at<float>(0,j)=cloud->points[horiInd[i][j]].x-cp.x;
				pointsInlier.at<float>(1,j)=cloud->points[horiInd[i][j]].y-cp.y;
				pointsInlier.at<float>(2,j)=cloud->points[horiInd[i][j]].z-cp.z;
				
			}
			horiPlanes_fi.push_back(hp);
			cv::Mat u,w,v;
			SVD::compute(pointsInlier, w, u, v);
			hn.x=u.at<float>(0,2);
			hn.y=u.at<float>(1,2);
			hn.z=u.at<float>(2,2);
			horiNormals_fi.push_back(hn);
		}
	}
	t[10]=clock();
	cout<<"horiPlanes_fi number: "<<horiPlanes_fi.size()<<endl;
	/*for (int i=0; i<horiPlanes_fi.size(); i++)
	{
		cout<<horiPlanes_fi[i].size()<<endl;
	}*/
	cout<<"horiPlanes number: "<<horiPlanes.size()<<endl;
	/*for (int i=0; i<horiPlanes.size(); i++)
	{
		cout<<horiPlanes[i].size()<<endl;
	}*/

	/*//write the running time into txt file
	char outPath3[100];
	sprintf(outPath3, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/time.txt", outputFile);
	ofstream myfile4(outPath3, ios::out);
	if(!myfile4)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		myfile4<<(double)(t[10]-t[2])/CLOCKS_PER_SEC<<" "<<"s"<<endl;
		myfile4.close();

	}*/

	//read color from "rgb.txt" which is generated by jet in matlab
	int RGBValue[15000][3];
	int rgbn=0;
    char inPath[100]="/home/victor/project_ws/src/ground_plane/src/rgb.txt";
	ifstream infile(inPath, ios::in);
	if(!infile)
	{
		cout<<"error reading rgb.txt!"<<endl;
	}
	else
	{
		for(int i=0; i<15000; i++)
		{
			infile>>RGBValue[i][0]>>RGBValue[i][1]>>RGBValue[i][2];
		}
		infile.close();
	}

	//write all the horizontal planes into ply file
	char outPath[100];
	sprintf(outPath, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/ransacPlane(num:%d).ply", outputFile, int(horiPlanes_fi.size()));
	ofstream myfile0(outPath, ios::out);
	if(!myfile0)
	{
		cout<<"error!"<<endl;
	}
	else
	{
		myfile0<<"ply"<<endl;
		myfile0<<"format ascii 1.0"<<endl;
		myfile0<<"element vertex "<<num_hori<<endl;
		myfile0<<"property float x"<<endl;
		myfile0<<"property float y"<<endl;
		myfile0<<"property float z"<<endl;
		myfile0<<"property uchar red"<<endl;
		myfile0<<"property uchar green"<<endl;
		myfile0<<"property uchar blue"<<endl;
		myfile0<<"element face 0"<<endl;
		myfile0<<"end_header"<<endl;
		for (int i=0; i<horiPlanes_fi.size(); i++)
		{
			for(int j=0; j<horiPlanes_fi[i].size(); j++)
			{
				myfile0<<horiPlanes_fi[i][j].x<<" "<<horiPlanes_fi[i][j].y<<" "<<horiPlanes_fi[i][j].z<<" "<<RGBValue[i][0]<<" "<<RGBValue[i][1]<<" "<<RGBValue[i][2]<<endl;
			}
		}
		myfile0.close();
	}

	/*//write the horizontal plane into ply file and store the normals of each plane
	for(int i=0; i<horiPlanes_fi.size(); i++)
	{
		//write the horizontal plane into ply file
		char outPath2[100];
		sprintf(outPath2, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/ransacPlane%d.ply", outputFile, i);
		ofstream myfile2(outPath2, ios::out);
		if(!myfile2)
		{
			cout<<"error!"<<endl;
		}
		else
		{
			myfile2<<"ply"<<endl;
			myfile2<<"format ascii 1.0"<<endl;
			myfile2<<"element vertex "<<horiPlanes_fi[i].size()<<endl;
			myfile2<<"property float x"<<endl;
			myfile2<<"property float y"<<endl;
			myfile2<<"property float z"<<endl;
			myfile2<<"property uchar red"<<endl;
			myfile2<<"property uchar green"<<endl;
			myfile2<<"property uchar blue"<<endl;
			myfile2<<"element face 0"<<endl;
			myfile2<<"end_header"<<endl;
			for(int j=0; j<horiPlanes_fi[i].size(); j++)
			{
				myfile2<<horiPlanes_fi[i][j].x<<" "<<horiPlanes_fi[i][j].y<<" "<<horiPlanes_fi[i][j].z<<" "<<"255 0 0"<<endl;
			}
			myfile2.close();
		}
		//store the normals and center point of each plane
		char outPath1[100];
		sprintf(outPath1, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s/normal%d.txt", outputFile, i);
		ofstream myfile3(outPath1, ios::out);
		if(!myfile3)
		{
			cout<<"error!"<<endl;
		}
		else
		{
			myfile3<<horiNormals_fi[i].x<<" "<<horiNormals_fi[i].y<<" "<<horiNormals_fi[i].z<<endl;
			myfile3<<horiCenters_fi[i].x<<" "<<horiCenters_fi[i].y<<" "<<horiCenters_fi[i].z<<endl;
		}
		myfile3.close();
	}*/
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
    ros::init(argc, argv, "FSPF");
	ros::NodeHandle n;
    string folderName="home_data_ascii";
	char mkFile[100];
	sprintf(mkFile, "/home/victor/project_ws/src/ground_plane/src/FSPF/%s", folderName.c_str());
	//cout<<mkFile<<endl;
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
		//cout<<files[i].c_str()<<endl;
		char inPath[100];
		sprintf(inPath, "/home/victor/project_ws/src/ground_plane/data/%s/%s", folderName.c_str(), files[i].c_str());
		char outPath[100];
		string fname;
		fname.assign(files[i].begin(), files[i].end()-4);
		sprintf(outPath, "%s/%s", folderName.c_str(), fname.c_str());
		FSPFHorizontalPlanes(inPath, outPath);
	}
    
    return 0;
}
