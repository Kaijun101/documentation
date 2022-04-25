---
hide_title: true
sidebar_label: kalibr installation guide and usage
---
## kalibr installation guide and usage:
  - Below is the complete kalibr camera calibration guide for intel realsense D435I.
  - References: https://github.com/chengguizi/basalt-mirror
  - References: https://github.com/matthewoots?after=Y3Vyc29yOnYyOpK5MjAyMS0wNS0wMlQyMTo0OTowNSswODowMM4VrFti&tab=repositories

### Complete kalibr guide for camera calibration:
 - Below is the sequences that is requried for the camera calibration:
	1. Installing common dependencies that is required for kalibr
	2. Printing the Aprilgrid (On A4)
	3. Generating the Aprilgrid.yaml file
	4. Generating IMU.yaml file
	5. Rosbag record (camera + imu)
	6. Obtaining Kalibr camera calibration
	7. Obtaining Kalibr camera + IMU calibration
	8. Converting the yaml file to jason file (Optional, depends if your VIO requires in jason file format)

### Installing common dependencies:
  - Copy the below scripts and save it inside a text file as sh file for ease of installation.
  ```bash
  #! /bin/sh
  #filename: kalibr.sh

  echo -e "$(tput setaf 185) \n Installing dependencies $(tput sgr 0) \n "
  sudo apt-get install -y python-setuptools python-rosinstall ipython libeigen3-dev libboost-all-dev doxygen libopencv-dev ros-$ROS_DISTRO-vision-opencv ros-$ROS_DISTRO-image-transport-plugins ros-melodic-cmake-modules software-properties-common libpoco-dev python-matplotlib python-scipy python-git python-pip ipython libtbb-dev libblas-dev liblapack-dev python-catkin-tools libv4l-dev python-igraph

  echo -e "$(tput setaf 185) \n Creating the requried workspace $(tput sgr 0) \n "
  cd ~/
  mkdir -p ~/kalibr_ws/src # Kalibr Workspace 
  mkdir ~/calibration # Data and bag files 

  echo -e "$(tput setaf 185) \n Setting up Catkin configuration and install additional dependencies $(tput sgr 0) \n "
  cd ~/kalibr_ws 
  catkin config --extend /opt/ros/melodic --merge-devel -DCMAKE_BUILD_TYPE=Release 
  sudo pip install Pillow 
  sudo apt-get -y install libv4l-dev 
  cd ~/kalibr_ws/src 
  git clone https://github.com/Kaijun101/kalibr.git
  echo -e "$(tput setaf 185) \n Building the packages $(tput sgr 0) \n "
  cd .. 
  catkin build -DCMAKE_BUILD_TYPE=Release -j2 

  echo -e "$(tput setaf 185) \n Sourcing the workspace $(tput sgr 0) \n "
  source devel/setup.bash 

  echo -e "$(tput setaf 185) \n kalibr installation completed $(tput sgr 0) \n "

  echo -e "$(tput setaf 185) \n Please open the kalibr.sh file to read the other required code for part 2 $(tput sgr 0) \n "
  ```

### Printing of a Aprilgird for kalibr calibration:
  - Edit the [NUM_COLS], [NUM_ROWS], [TAG_WIDTH_M] and [TAG_SPACING_PERCENT] accordingly to fit it inside a A4 paper for calibration. 
  - The Aprilgrid is required to have more then 15 apriltags for a better camera calibration result.
  ```bash
  cd ~/kalibr_ws 
  source devel/setup.bash
  kalibr_create_target_pdf --type apriltag --nx [NUM_COLS] --ny [NUM_ROWS] --tsize [TAG_WIDTH_M] --tspace [TAG_SPACING_PERCENT] #edit according to fit inside the A4 paper
  ```

### Generating Aprilgrid.yaml file:
  - All the value here are in metre and is the actual measurement taken from your printed Aprilgrid.
  ```bash
  target_type: 'aprilgrid'             #gridtype
  tagCols: 5                           #number of apriltags
  tagRows: 5                           #number of apriltags
  tagSize: 0.027                       #size of apriltag, edge to edge [m]
  tagSpacing: 0.2962962963             #ratio of space between tags to tagSize, for our apriltag spacing = 0.029 (Formula=spacing/size) #space between tags=0.008
  low_id: 25
  ```

### Example of Aprilgrid.jason file (Some VIO may requried jason file format):
  ```bash
	{
	  "tagCols": 5,
	  "tagRows": 5,
	  "tagSize": 0.027,
	  "tagSpacing": 0.2962962963
	}
  ```

### Generating IMU results:
  - Copy the information and save as a imu yaml file.
  - Below the is most accurate imu.yaml file that we obtain for realsense D435I.
  - Make sure to check if your rostopic and update rate is correct. (Use rostopic hz /camera/hz to check the update rate)
  ```bash
	#Accelerometers
	accelerometer_noise_density: 0.001865        #Noise density (continuous-time)
	accelerometer_random_walk:   0.0002          #Bias random walk

	#Gyroscopes
	gyroscope_noise_density:     0.0018685       #Noise density (continuous-time)
	gyroscope_random_walk:       4.0e-06         #Bias random walk

	rostopic:                    /camera/imu     #the IMU ROS topic
	update_rate:                 200.0           #Hz (for discretization of the values above)
  ```

### Recording the Rosbag file:
  - Make sure you off the emitter before you rosbag record the video. Also, change the frame rate to 6hz and camera height and width to (640 x 480).
  - Before recording, make sure to do all the required camera calibration for the realsense (imu calibration, depth image calibration and camera calibration)
   ![image](https://user-images.githubusercontent.com/90491438/165130384-4ec1a1d2-94cd-4522-9454-0e85f6102d24.png)
   ![image](https://user-images.githubusercontent.com/90491438/165130523-bc2122eb-4ff3-43b7-b70b-2146d7e30d90.png)
   ![image](https://user-images.githubusercontent.com/90491438/165130755-9b8bfd01-aa67-4d1e-bc2f-a99a8b5ae243.png)


  - Adjust the exposure and gain and ensure that when you shake the camera slighly, the image is still very clear.
  - Below is the example of recording the rosbag file for realsense D435I.
  - camera-imu-data.bag: Rosbag file name.
  - The rest are the topic that is required (camera video + IMU).
  ```bash
  rosbag record -O camera-imu-data.bag /camera/imu /camera/infra1/image_rect_raw /camera/infra2/image_rect_raw
  ```

#### Recording a compress Rosbag file:
  - Below is the example of recording the compress rosbag file for realsense D435I.
  - This is required as max recording is up to 4gb and 
  ```bash
  rosbag record -- lz4 camera-imu-data.bag /camera/imu /camera/infra1/image_rect_raw /camera/infra2/image_rect_raw
  ```
#### Run the below code to decompress the rosbag file:
  ```bash
  rosbag decompress camera-imu-data.bag
  ```

### Getting the Kalibr camera result:
  - Things you need: Aprilgrid.yaml and rosbag file (Camera + imu).
  ```bash
  cd ~/kalibr_ws 
  source devel/setup.bash
  kalibr_calibrate_cameras --target <aprilgrid_yaml_file_location> --bag <rosbag_file_location> --models pinhole-radtan pinhole-radtan --topics /camera/infra1/image_rect_raw /camera/infra2/image_rect_raw
  ```

### Getting the Kalibr camera result with IMU result:
  - Things you need: Aprilgrid.yaml, rosbag file (Camera + IMU) and imu.yaml.
  ```bash
  cd ~/kalibr_ws 
  source devel/setup.bash
  kalibr_calibrate_imu_camera --target <aprilgrid_yaml_file_location> --bag <rosbag_file_location> --cam <Kalibr_camera_calibration_result_obtain_from_above> --imu <IMU_yaml_file_location>
  ```
### Review of kalibr result
  - Below is the image of one section of kalibr result, the red line represent the ideal range. If the reading is too far out of the ideal range, you should redo the whole camera calibration. Alternatively, you can try with the VIO to see how much drift occurs with the new camera calibration result.
  - VIO drift occurs due to several reason such poor camera calibration, wrong coordinate frame and improper lighting.
![image](https://user-images.githubusercontent.com/90491438/165131004-2dc8bb1f-051c-4447-905f-97d79bf9e1c3.png)


### Converting kalibr yaml file to jason file (Requried file format for basalt VIO):
  - As some VIO requires the calibration to be in jason format, below is a python scripts that will convert the camera calibration from yaml to jason format. However, it requries scipy and sophus to work. 
#### Install scipy:
  ```bash
  sudo apt-get install python-scipy 
  pip3 install scipy
  ```

#### Install sophus:
  ```bash
  git clone https://github.com/pybind/pybind11.git
  cd pybind11
  mkdir build && cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=../install/
  make -j8
  make install
  pybind11_DIR=$PWD/../install/share/cmake/pybind11/ pip3 install --user sophuspy
  pip3 install sophuspy
  ```

  - Copy the below scripts and save it inside a text file as a pinhole_kalibr_rs.py file.
  - Make sure to edited the "distortion_coeffs" that is inside this python file that is obtain from the kalibr result before running the python scripts.
  ```bash
	#!/usr/bin/env python3

	import yaml
	import argparse
	import numpy as np
	from string import Template
	from scipy.spatial.transform import Rotation as R
	import sophus as sp

	# Obtain the Path
	parser = argparse.ArgumentParser(description='Convert Kalibr Calibration to Basalt-like parameters')
	parser.add_argument('yaml', type=str, help='Kalibr Yaml file path')
	parser.add_argument('output_name', type=str, help='Output name of the json file')
	args = parser.parse_args()
	print(args.yaml)
	#tis param
		        # "px": 0.03,
		        # "py": 0,
		        # "pz": 0,
		        # "qx": 0,
		        # "qy": 0,
		        # "qz": 1,
		        # "qw": 0

	calib_template = Template('''{
	    "value0": {
		"T_imu_cam": [
		    {
		        "px": $px0,
		        "py": $py0,
		        "pz": $pz0,
		        "qx": $qx0,
		        "qy": $qy0,
		        "qz": $qz0,
		        "qw": $qw0
		    },
		    {
		        "px": $px1,
		        "py": $py1,
		        "pz": $pz1,
		        "qx": $qx1,
		        "qy": $qy1,
		        "qz": $qz1,
		        "qw": $qw1
		    }
		],
		"intrinsics": [
		    {
		        "camera_type": "pinhole",
		        "intrinsics": {
		            "fx": $fx0,
		            "fy": $fy0,
		            "cx": $cx0,
		            "cy": $cy0
		        }
		    },
		    {
		        "camera_type": "pinhole",
		        "intrinsics": {
		            "fx": $fx1,
		            "fy": $fy1,
		            "cx": $cx1,
		            "cy": $cy1
		        }
		    }
		],
		"resolution": [
		    [
		        $rx,
		        $ry
		    ],
		    [
		        $rx,
		        $ry
		    ]
		],
		"vignette": [],
		"calib_accel_bias": [
		    0.0,
		    0.0,
		    0.0,
		    0.0,
		    0.0,
		    0.0,
		    0.0,
		    0.0,
		    0.0
		],
		"calib_gyro_bias": [
		    0.0,
		    0.0,
		    0.0,
		    0.0,
		    0.0,
		    0.0,
		    0.0,
		    0.0,
		    0.0,
		    0.0,
		    0.0,
		    0.0
		],
		"imu_update_rate": $imu_rate,
		"accel_noise_std": [0.016, 0.016, 0.016],
		"gyro_noise_std": [0.000282, 0.000282, 0.000282],
		"accel_bias_std": [0.001, 0.001, 0.001],
		"gyro_bias_std": [0.0001, 0.0001, 0.0001],
		"cam_time_offset_ns": 0,
		"distortion_coeffs": [0.0046918871601571875, -0.005166682490625372, -0.0012857120748903227,
	    -0.0024133265867770856]
	    }
	}
	''')

	stream = open(args.yaml, 'r')
	# stream = open("/media/nvidia/SD/catkin_ws/src/basalt-mirror/data/tis_23/camchain-imucam-2020-08-08-16-00-21.yaml", 'r')


	f = yaml.safe_load(stream)
	stream.close()

	T_c1_c0 = sp.SE3(f['cam1']['T_cn_cnm1'])

	print('camera 0 in camera 1 transformation:')
	print(T_c1_c0)

	print('camera 0 in imu transformation')
	# assume IMU is in NWU frame and is mounting facing forward
	# assume the two cameras are mounted forward too. frame right-down-forward
	R_imu_c0 = sp.SO3([ [ 1, 0, 0],
		            [0, 1, 0],
		            [ 0, 0, 1]])
	t_imu_c0 = [0, 0 , 0]
	T_imu_c0 = sp.SE3(R_imu_c0.matrix(),t_imu_c0)
	print(T_imu_c0)

	q_imu_c0 = R.from_matrix(R_imu_c0.matrix()).as_quat()

	T_imu_c1 = T_imu_c0 * T_c1_c0.inverse()
	print('camera 1 in imu transformation')
	print(T_imu_c1)

	t_imu_c1 = T_imu_c1.translation()

	q_imu_c1 = R.from_matrix(T_imu_c1.rotationMatrix()).as_quat()

	distort_0 = f['cam0']['distortion_coeffs']
	distort_1 = f['cam1']['distortion_coeffs']

	intrinsics_0 = f['cam0']['intrinsics']
	intrinsics_1 = f['cam1']['intrinsics']

	resolution_0 = f['cam0']['resolution']
	resolution_1 = f['cam1']['resolution']

	# transformations are all respect to imu frame
	values = {'px0':  t_imu_c0[0] , 'py0':  t_imu_c0[1]  ,'pz0':  t_imu_c0[2]  ,
		    'px1':  t_imu_c1[0] , 'py1':  t_imu_c1[1] , 'pz1':  t_imu_c1[2]  ,
		    'qx0':  q_imu_c0[0] , 'qy0':  q_imu_c0[1] , 'qz0':  q_imu_c0[2] , 'qw0':  q_imu_c0[3] ,
		    'qx1':  q_imu_c1[0] , 'qy1':  q_imu_c1[1] , 'qz1':  q_imu_c1[2] , 'qw1':  q_imu_c1[3] ,
		    'fx0': intrinsics_0[0], 'fy0': intrinsics_0[1], 'cx0': intrinsics_0[2], 'cy0': intrinsics_0[3], 
		    'fx1': intrinsics_1[0], 'fy1': intrinsics_1[1], 'cx1': intrinsics_1[2], 'cy1': intrinsics_1[3], 
		    'rx': resolution_0[0], 'ry': resolution_0[1],
		    'imu_rate' : 200.0}


	calib = calib_template.substitute(values)
	print(calib)

	with open('./'+ args.output_name + '.json', 'w') as stream2:
	    stream2.write(calib)
  ```

### Running the python script:
  ```bash
  chmod 755 ./pinhole_kalibr_rs.py
  ./pinhole_kalibr_rs.py <kalibr_calibration_result_file_location> <Name_of_your_output_file>
  ```


