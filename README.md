# Oriented-Point-Sampling
## Introduction
This work is based on our ICRA'19 paper: "Oriented Point Sampling for Plane Detection in Unorganized Point Clouds.". You can find arXiv version of the paper here: https://arxiv.org/abs/1905.02553.
## Citation
If you find our work useful in your research, please consider citing:

        @article{sun2019oriented,
          title={Oriented point sampling for plane detection in unorganized point clouds},
          author={Sun, Bo and Mordohai, Philippos},
          journal={2019 International Conference on Robotics and Automation (ICRA)},
          pages={2917--2923},
          year={2019},
          organization={IEEE}
        }
## Prerequisite
(1) Ubuntu 16.04  
(2) Robotic Operating System(ROS): https://www.ros.org/  
(3) Point Cloud Library(PCL): https://pointclouds.org/  
(4) OpenCV: https://opencv.org/  
(5) SUN RGB-D dataset: https://rgbd.cs.princeton.edu/
## Usage
(1) Go through "1.1 Beginner Level" (http://wiki.ros.org/ROS/Tutorials) to make sure your ROS is installed correctly.  
(2) Follow "Creating a ROS package" (http://wiki.ros.org/ROS/Tutorials/CreatingPackage) to make a new ROS package.  
(3) Put "FSPF.cpp", "ops.cpp" and "read_points_gt.cpp" in the "src" folder in your package which is created in (2).  
(4) Follow "Building a ROS Package" (http://wiki.ros.org/ROS/Tutorials/BuildingPackages) to run the code.  
"FSPF.cpp": Our implementation of FSPF from RSS'11 paper "Fast sampling plane filtering, polygon construction and merging from depth images." (http://www.cs.cmu.edu/~coral/projects/cobot/papers/PlaneFiltering.pdf)  
"ops.cpp": OPS plane detection method in our paper.  
"read_points_gt.cpp": Ground truth plane generation method in our paper.
## License
Our code is released under MIT License (see LICENSE file for details).
