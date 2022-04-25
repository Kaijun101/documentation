---
hide_title: true
sidebar_label: kalibr installation guide and usage
---
## kalibr installation guide and usage:
  - Below is the complete kalibr camera calibration guide for intel realsense D435I.
  - Credit: https://github.com/chengguizi/basalt-mirror
  - Credit: https://github.com/matthewoots?after=Y3Vyc29yOnYyOpK5MjAyMS0wNS0wMlQyMTo0OTowNSswODowMM4VrFti&tab=repositories

### Complete kalibr guide for camera calibration:
 - Below is the sequences that is requried for the camera calibration:
	1. Installing common dependencies that is required for kalibr
	2. Printing the Aprilgrid (On A4)
	3. Generating the Aprilgrid.yaml file
	4. Generating IMU.yaml file
	5. Realsense camera calibration (To ensure higher accuracy)
	6. Rosbag record (camera + imu)
	7. Obtaining Kalibr camera calibration
	8. Obtaining Kalibr camera + IMU calibration
	9. Converting the yaml file to jason file (Optional, depends if your VIO requires in jason file format)

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
### Realsense camera calibration (To ensure higher accuracy)
#### Realsense camera IMU calibration
  - Follow the steps for the Realsense D435I IMU calibration. References: https://www.intelrealsense.com/wp-content/uploads/2019/07/Intel_RealSense_Depth_D435i_IMU_Calibration.pdf
  - Below is the "rs-imu-calibration.py" scripts that you would requried for the camera IMU calibration.
  ```bash 
	#!/usr/bin/python
	from __future__ import print_function
	import numpy as np
	import sys
	import json
	import ctypes
	import os
	import binascii
	import struct
	import pyrealsense2 as rs
	import ctypes
	import time
	import enum
	import threading

	# L515
	READ_TABLE  = 0x43     # READ_TABLE 0x243 0
	WRITE_TABLE = 0x44     # WRITE_TABLE 0 <table>

	# L515 minimum firmware version required to support IMU calibration
	L515_FW_VER_REQUIRED = '01.04.01.00'

	is_data = None
	get_key = None
	if os.name == 'posix':
	    import select
	    import tty
	    import termios

	    is_data = lambda : select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
	    get_key = lambda : sys.stdin.read(1)

	elif os.name == 'nt':
	    import msvcrt
	    is_data = msvcrt.kbhit
	    get_key = lambda : msvcrt.getch()

	else:
	    raise Exception('Unsupported OS: %s' % os.name)

	if sys.version_info[0] < 3:
	    input = raw_input

	max_float = struct.unpack('f',b'\xff\xff\xff\xff')[0]
	max_int = struct.unpack('i',b'\xff\xff\xff\xff')[0]
	max_uint8 = struct.unpack('B', b'\xff')[0]

	g = 9.80665 # SI Gravity page 52 of https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication330e2008.pdf

	COLOR_RED   = "\033[1;31m"  
	COLOR_BLUE  = "\033[1;34m"
	COLOR_CYAN  = "\033[1;36m"
	COLOR_GREEN = "\033[0;32m"
	COLOR_RESET = "\033[0;0m"
	COLOR_BOLD    = "\033[;1m"
	COLOR_REVERSE = "\033[;7m"

	def int_to_bytes(num, length=4, order='big'):
	    res = bytearray(length)
	    for i in range(length):
		res[i] = num & 0xff
		num >>= 8
	    if num:
		raise OverflowError("Number {} doesn't fit into {} bytes.".format(num, length))
	    if order == 'little':
		res.reverse()
	    return res


	def bytes_to_uint(bytes_array, order='little'):
	    bytes_array = list(bytes_array)
	    bytes_array.reverse()
	    if order == 'little':
		return struct.unpack('>i', struct.pack('BBBB', *([0] * (4 - len(bytes_array))) + bytes_array))[0] & 0xffffffff
	    else:
		return struct.unpack('>i', struct.pack('BBBB', *([0] * (4 - len(bytes_array))) + bytes_array))[0] & 0xffffffff


	class imu_wrapper:
	    class Status(enum.Enum):
		idle = 0,
		rotate = 1,
		wait_to_stable = 2,
		collect_data = 3

	    def __init__(self):
		self.pipeline = None
		self.imu_sensor = None
		self.status = self.Status(self.Status.idle)     # 0 - idle, 1 - rotate to position, 2 - wait to stable, 3 - pick data
		self.thread = threading.Condition()
		self.step_start_time = time.time()
		self.time_to_stable = 3
		self.time_to_collect = 2
		self.samples_to_collect = 1000
		self.rotating_threshold = 0.1
		self.moving_threshold_factor = 0.1
		self.collected_data_gyro = []
		self.collected_data_accel = []
		self.callback_lock = threading.Lock()
		self.max_norm = np.linalg.norm(np.array([0.5, 0.5, 0.5]))
		self.line_length = 20
		self.is_done = False
		self.is_data = False

	    def escape_handler(self):
		self.thread.acquire()
		self.status = self.Status.idle
		self.is_done = True
		self.thread.notify()
		self.thread.release()
		sys.exit(-1)

	    def imu_callback(self, frame):
		if not self.is_data:
		    self.is_data = True

		with self.callback_lock:
		    try:
			if is_data():
			    c = get_key()
			    if c == '\x1b':         # x1b is ESC
				self.escape_handler()

			if self.status == self.Status.idle:
			    return
			pr = frame.get_profile()
			data = frame.as_motion_frame().get_motion_data()
			data_np = np.array([data.x, data.y, data.z])
			elapsed_time = time.time() - self.step_start_time

			## Status.collect_data
			if self.status == self.Status.collect_data:
			    sys.stdout.write('\r %15s' % self.status)
			    part_done = len(self.collected_data_accel) / float(self.samples_to_collect)
			    # sys.stdout.write(': %-3.1f (secs)' % (self.time_to_collect - elapsed_time))

			    color = COLOR_GREEN
			    if pr.stream_type() == rs.stream.gyro:
				self.collected_data_gyro.append(np.append(frame.get_timestamp(), data_np))
				is_moving = any(abs(data_np) > self.rotating_threshold)
			    else:
				is_in_norm = np.linalg.norm(data_np - self.crnt_bucket) < self.max_norm
				if is_in_norm:
				    self.collected_data_accel.append(np.append(frame.get_timestamp(), data_np))
				else:
				    color = COLOR_RED
				is_moving = abs(np.linalg.norm(data_np) - g) / g > self.moving_threshold_factor

				sys.stdout.write(color)
				sys.stdout.write('['+'.'*int(part_done*self.line_length)+' '*int((1-part_done)*self.line_length) + ']')
				sys.stdout.write(COLOR_RESET)

			    if is_moving:
				print('WARNING: MOVING')
				self.status = self.Status.rotate
				return

			    # if elapsed_time > self.time_to_collect:
			    if part_done >= 1:
				self.status = self.Status.collect_data
				sys.stdout.write('\n\nDirection data collected.')
				self.thread.acquire()
				self.status = self.Status.idle
				self.thread.notify()
				self.thread.release()
				return

			if pr.stream_type() == rs.stream.gyro:
			    return
			sys.stdout.write('\r %15s' % self.status)
			crnt_dir = np.array(data_np) / np.linalg.norm(data_np)
			crnt_diff = self.crnt_direction - crnt_dir
			is_in_norm = np.linalg.norm(data_np - self.crnt_bucket) < self.max_norm               

			## Status.rotate
			if self.status == self.Status.rotate:
			    sys.stdout.write(': %35s' % (np.array2string(crnt_diff,  precision=4, suppress_small=True)))
			    sys.stdout.write(': %35s' % (np.array2string(abs(crnt_diff) < 0.1)))
			    if is_in_norm:
				self.status = self.Status.wait_to_stable
				sys.stdout.write('\r'+' '*90)
				self.step_start_time = time.time()
				return

			## Status.wait_to_stable
			if self.status == self.Status.wait_to_stable:
			    sys.stdout.write(': %-3.1f (secs)' % (self.time_to_stable - elapsed_time))
			    if not is_in_norm:
				self.status = self.Status.rotate
				return
			    if elapsed_time > self.time_to_stable:
				self.collected_data_gyro = []
				self.collected_data_accel = []
				self.status = self.Status.collect_data
				self.step_start_time = time.time()
				return
			return
		    except Exception as e:
			print('ERROR?' + str(e))
			self.thread.acquire()
			self.status = self.Status.idle
			self.thread.notify()
			self.thread.release()

	    def get_measurements(self, buckets, bucket_labels):
		measurements = []
		print('-------------------------')
		print('*** Press ESC to Quit ***')
		print('-------------------------')
		for bucket,bucket_label in zip(buckets, bucket_labels):
		    self.crnt_bucket = np.array(bucket)
		    self.crnt_direction = np.array(bucket) / np.linalg.norm(np.array(bucket))
		    print('\nAlign to direction: ', self.crnt_direction, ' ', bucket_label)
		    self.status = self.Status.rotate
		    self.thread.acquire()
		    while (not self.is_done and self.status != self.Status.idle):
			self.thread.wait(3)
			if not self.is_data:
			    raise Exception('No IMU data. Check connectivity.')
		    if self.is_done:
			raise Exception('User Abort.')
		    measurements.append(np.array(self.collected_data_accel))
		return np.array(measurements), np.array(self.collected_data_gyro)

	    def enable_imu_device(self, serial_no):
		self.pipeline = rs.pipeline()
		cfg = rs.config()
		cfg.enable_device(serial_no)
		try:
		    self.pipeline.start(cfg)
		except Exception as e:
		    print('ERROR: ', str(e))
		    return False

		# self.sync_imu_by_this_stream = rs.stream.any
		active_imu_profiles = []

		active_profiles = dict()
		self.imu_sensor = None
		for sensor in self.pipeline.get_active_profile().get_device().sensors:
		    for pr in sensor.get_stream_profiles():
			if pr.stream_type() == rs.stream.gyro and pr.format() == rs.format.motion_xyz32f:
			    active_profiles[pr.stream_type()] = pr
			    self.imu_sensor = sensor
			if pr.stream_type() == rs.stream.accel and pr.format() == rs.format.motion_xyz32f:
			    active_profiles[pr.stream_type()] = pr
			    self.imu_sensor = sensor
		    if self.imu_sensor:
			break
		if not self.imu_sensor:
		    print('No IMU sensor found.')
		    return False
		print('\n'.join(['FOUND %s with fps=%s' % (str(ap[0]).split('.')[1].upper(), ap[1].fps()) for ap in active_profiles.items()]))
		active_imu_profiles = list(active_profiles.values())
		if len(active_imu_profiles) < 2:
		    print('Not all IMU streams found.')
		    return False
		self.imu_sensor.stop()
		self.imu_sensor.close()
		self.imu_sensor.open(active_imu_profiles)
		self.imu_start_loop_time = time.time()
		self.imu_sensor.start(self.imu_callback)

		# Make the device use the original IMU values and not already calibrated:
		if self.imu_sensor.supports(rs.option.enable_motion_correction):
		    self.imu_sensor.set_option(rs.option.enable_motion_correction, 0)
		return True

	class CHeader:
	    def __init__(self, version, table_type):
		self.buffer = np.ones(16, dtype=np.uint8) * 255
		self.buffer[0] = int(version[0], 16)
		self.buffer[1] = int(version[1], 16)
		self.buffer.dtype=np.uint16
		self.buffer[1] = int(table_type, 16)

	    def size(self):
		return 16

	    def set_data_size(self, size):
		self.buffer.dtype=np.uint32
		self.buffer[1] = size

	    def set_crc32(self, crc32):
		self.buffer.dtype=np.uint32
		self.buffer[3] = crc32 % (1<<32)    # convert from signed to unsigned 32 bit

	    def get_buffer(self):
		self.buffer.dtype=np.uint8
		return self.buffer


	def bitwise_int_to_float(ival):
	    return struct.unpack('f', struct.pack('i', ival))[0]

	def bitwise_float_to_int(fval):
	    return struct.unpack('i', struct.pack('f', fval))[0]

	def parse_buffer(buffer):
	    cmd_size = 24
	    header_size = 16

	    buffer.dtype=np.uint32    
	    tab1_size = buffer[3]
	    buffer.dtype=np.uint8
	    print('tab1_size (all_data): ', tab1_size)

	    tab1 = buffer[cmd_size:cmd_size+tab1_size]  # 520 == epprom++
	    tab1.dtype=np.uint32
	    tab2_size = tab1[1]
	    tab1.dtype=np.uint8
	    print('tab2_size (calibration_table): ', tab2_size)

	    tab2 = tab1[header_size:header_size+tab2_size] # calibration table
	    tab2.dtype=np.uint32
	    tab3_size = tab2[1]
	    tab2.dtype=np.uint8
	    print('tab3_size (calibration_table): ', tab3_size)

	    tab3 = tab2[header_size:header_size+tab3_size]  # D435 IMU Calib Table
	    tab3.dtype=np.uint32
	    tab4_size = tab3[1]
	    tab3.dtype=np.uint8
	    print('tab4_size (D435_IMU_Calib_Table): ', tab4_size)

	    tab4 = tab3[header_size:header_size+tab4_size]  # calibration data
	    return tab1, tab2, tab3, tab4

	def get_IMU_Calib_Table(X, product_line):
	    version = ['0x02', '0x01']
	    table_type = '0x20'

	    if product_line == 'L500':
		version = ['0x05', '0x01']
		table_type = '0x243'

	    header = CHeader(version, table_type)

	    header_size = header.size()
	    data_size = 37*4 + 96
	    size_of_buffer = header_size + data_size    # according to table "D435 IMU Calib Table" here: https://user-images.githubusercontent.com/6958867/50902974-20507500-1425-11e9-8ca5-8bd2ac2d0ea1.png
	    assert(size_of_buffer % 4 == 0)
	    buffer = np.ones(size_of_buffer, dtype=np.uint8) * 255

	    use_extrinsics = False
	    use_intrinsics = True

	    data_buffer = np.ones(data_size, dtype=np.uint8) * 255
	    data_buffer.dtype = np.float32

	    data_buffer[0] = bitwise_int_to_float(np.int32(int(use_intrinsics)) << 8 | 
						  np.int32(int(use_extrinsics)))

	    intrinsic_vector = np.zeros(24, dtype=np.float32)
	    intrinsic_vector[:9] = X[:3,:3].T.flatten()
	    intrinsic_vector[9:12] = X[:3,3]
	    intrinsic_vector[12:21] = X[3:,:3].flatten()
	    intrinsic_vector[21:24] = X[3:,3]    

	    data_buffer[13:13+X.size] = intrinsic_vector
	    data_buffer.dtype = np.uint8

	    header.set_data_size(data_size)

	    header.set_crc32(binascii.crc32(data_buffer))
	    buffer[:header_size] = header.get_buffer()
	    buffer[header_size:] = data_buffer
	    return buffer


	def get_calibration_table(d435_imu_calib_table):
	    version = ['0x02', '0x00']
	    table_type = '0x20'

	    header = CHeader(version, table_type)

	    d435_imu_calib_table_size = d435_imu_calib_table.size
	    sn_table_size = 32
	    data_size = d435_imu_calib_table_size + sn_table_size

	    header_size = header.size()
	    size_of_buffer = header_size + data_size    # according to table "D435 IMU Calib Table" in "https://sharepoint.ger.ith.intel.com/sites/3D_project/Shared%20Documents/Arch/D400/FW/D435i_IMU_Calibration_eeprom_0_52.xlsx"
	    assert(size_of_buffer % 4 == 0)
	    buffer = np.ones(size_of_buffer, dtype=np.uint8) * 255

	    data_buffer = np.ones(data_size, dtype=np.uint8) * 255
	    data_buffer[:d435_imu_calib_table_size] = d435_imu_calib_table

	    header.set_data_size(data_size)
	    header.set_crc32(binascii.crc32(data_buffer))

	    buffer[:header_size] = header.get_buffer()
	    buffer[header_size:header_size+data_size] = data_buffer
	    return buffer

	def get_eeprom(calibration_table):
	    version = ['0x01', '0x01']
	    table_type = '0x09'

	    header = CHeader(version, table_type)

	    DC_MM_EEPROM_SIZE = 520
	    # data_size = calibration_table.size

	    header_size = header.size()
	    size_of_buffer = DC_MM_EEPROM_SIZE
	    data_size = size_of_buffer - header_size 
	    # size_of_buffer = header_size + data_size

	    assert(size_of_buffer % 4 == 0)
	    buffer = np.ones(size_of_buffer, dtype=np.uint8) * 255

	    header.set_data_size(data_size)
	    buffer[header_size:header_size+calibration_table.size] = calibration_table
	    header.set_crc32(binascii.crc32(buffer[header_size:]))

	    buffer[:header_size] = header.get_buffer()

	    return buffer

	def write_eeprom_to_camera(eeprom, serial_no=''):
	    # DC_MM_EEPROM_SIZE = 520
	    DC_MM_EEPROM_SIZE = eeprom.size
	    DS5_CMD_LENGTH = 24

	    MMEW_Cmd_bytes = b'\x14\x00\xab\xcd\x50\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'


	    buffer = np.ones([DC_MM_EEPROM_SIZE + DS5_CMD_LENGTH, ], dtype = np.uint8) * 255
	    cmd = np.array(struct.unpack('I'*6, MMEW_Cmd_bytes), dtype=np.uint32)
	    cmd.dtype = np.uint16
	    cmd[0] += DC_MM_EEPROM_SIZE
	    cmd.dtype = np.uint32
	    cmd[3] = DC_MM_EEPROM_SIZE  # command 1 = 0x50
					# command 2 = 0
					# command 3 = size
	    cmd.dtype = np.uint8
	    buffer[:len(cmd)] = cmd
	    buffer[len(cmd):len(cmd)+eeprom.size] = eeprom

	    debug = get_debug_device(serial_no)
	    if not debug:
		print('Error getting RealSense Device.')
		return
	    # tab1, tab2, tab3, tab4 = parse_buffer(buffer)

	    rcvBuf = debug.send_and_receive_raw_data(bytearray(buffer))
	    if rcvBuf[0] == buffer[4]:
		print('SUCCESS: saved calibration to camera.')
	    else:
		print('FAILED: failed to save calibration to camera.')
		print(rcvBuf)


	def get_debug_device(serial_no):
	    ctx = rs.context()
	    devices = ctx.query_devices()
	    found_dev = False
	    for dev in devices:
		if len(serial_no) == 0 or serial_no == dev.get_info(rs.camera_info.serial_number):
		    found_dev = True
		    break
	    if not found_dev:
		print('No RealSense device found' + str('.' if len(serial_no) == 0 else ' with serial number: '+serial_no))
		return 0

	    # print(a few basic information about the device)
	    print('  Device PID: ',  dev.get_info(rs.camera_info.product_id))
	    print('  Device name: ',  dev.get_info(rs.camera_info.name))
	    print('  Serial number: ',  dev.get_info(rs.camera_info.serial_number))
	    print('  Firmware version: ',  dev.get_info(rs.camera_info.firmware_version))
	    debug = rs.debug_protocol(dev)
	    return debug

	def check_X(X, accel, show_graph):
	    fdata = np.apply_along_axis(np.dot, 1, accel, X[:3,:3]) - X[3,:]
	    norm_data = (accel**2).sum(axis=1)**(1./2)
	    norm_fdata = (fdata**2).sum(axis=1)**(1./2)
	    if show_graph:
		import pylab
		pylab.plot(norm_data, '.b')
		#pylab.hold(True)
		pylab.plot(norm_fdata, '.g')
		pylab.show()
	    print ('norm (raw data  ): %f' % np.mean(norm_data))
	    print ('norm (fixed data): %f' % np.mean(norm_fdata), "A good calibration will be near %f" % g)

	def l500_send_command(dev, op_code, param1=0, param2=0, param3=0, param4=0, data=[], retries=1):

	    for i in range(retries):
		try:
		    debug_device = rs.debug_protocol(dev)
		    gvd_command_length = 0x14 + len(data)
		    magic_number1 = 0xab
		    magic_number2 = 0xcd

		    buf = bytearray()
		    buf += bytes(int_to_bytes(gvd_command_length, 2))
		    #buf += bytes(int_to_bytes(0, 1))
		    buf += bytes(int_to_bytes(magic_number1, 1))
		    buf += bytes(int_to_bytes(magic_number2, 1))
		    buf += bytes(int_to_bytes(op_code))
		    buf += bytes(int_to_bytes(param1))
		    buf += bytes(int_to_bytes(param2))
		    buf += bytes(int_to_bytes(param3))
		    buf += bytes(int_to_bytes(param4))
		    buf += bytearray(data)
		    l = list(buf)
		    res = debug_device.send_and_receive_raw_data(buf)

		    if res[0] == op_code:
			res1 = res[4:]
			return res1
		    else:
			raise Exception("send_command return error", res[0])
		except:
		    if i < retries - 1:
			time.sleep(0.1)
		    else:
			raise

	def wait_for_rs_device(serial_no):
	    ctx = rs.context()

	    start = int(round(time.time() * 1000))
	    now = int(round(time.time() * 1000))

	    while now - start < 5000:
		devices = ctx.query_devices()
		for dev in devices:
		    pid = str(dev.get_info(rs.camera_info.product_id))
		    if len(serial_no) == 0 or serial_no == dev.get_info(rs.camera_info.serial_number):

			# print(a few basic information about the device)
			print('  Device PID: ',  dev.get_info(rs.camera_info.product_id))
			print('  Device name: ',  dev.get_info(rs.camera_info.name))
			print('  Serial number: ',  dev.get_info(rs.camera_info.serial_number))
			print('  Product Line: ',  dev.get_info(rs.camera_info.product_line))
			print('  Firmware version: ',  dev.get_info(rs.camera_info.firmware_version))

			return dev
		time.sleep(5)
		now = int(round(time.time() * 1000))
	    raise Exception('No RealSense device' + str('.' if len(serial_no) == 0 else ' with serial number: '+serial_no))


	def main():
	    if any([help_str in sys.argv for help_str in ['-h', '--help', '/?']]):
		print("Usage:", sys.argv[0], "[Options]")
		print
		print('[Options]:')
		print('-i : /path/to/accel.txt [/path/to/gyro.txt]')
		print('-s : serial number of device to calibrate.')
		print('-g : show graph of norm values - original values in blue and corrected in green.')
		print
		print('If -i option is given, calibration is done using previosly saved files')
		print('Otherwise, an interactive process is followed.')
		sys.exit(1)

	    try:
		accel_file = None
		gyro_file = None
		serial_no = ''
		show_graph = '-g' in sys.argv
		for idx in range(len(sys.argv)):
		    if sys.argv[idx] == '-i':
			accel_file = sys.argv[idx+1]
			if len(sys.argv) > idx+2 and not sys.argv[idx+2].startswith('-'):
			    gyro_file = sys.argv[idx+2]
		    if sys.argv[idx] == '-s':
			serial_no = sys.argv[idx+1]

		print('waiting for realsense device...')

		dev = wait_for_rs_device(serial_no)

		product_line = dev.get_info(rs.camera_info.product_line)

		if product_line == 'L500':
		    print('checking minimum firmware requirement ...')
		    fw_version = dev.get_info(rs.camera_info.firmware_version)
		    if fw_version < L515_FW_VER_REQUIRED:
			raise Exception('L515 requires firmware ' + L515_FW_VER_REQUIRED + " or later to support IMU calibration. Please upgrade firmware and try again.")
		    else:
			print('  firmware ' + fw_version + ' passed check.')

		buckets = [[0, -g,  0], [ g,  0, 0],
			[0,  g,  0], [-g,  0, 0],
			[0,  0, -g], [ 0,  0, g]]

		# all D400 and L500 cameras with IMU equipped with a mounting screw at the bottom of the device
		# when device is in normal use position upright facing out, mount screw is pointing down, aligned with positive Y direction in depth coordinate system
		# IMU output on each of these devices is transformed into the depth coordinate system, i.e.,
		# looking from back of the camera towards front, the positive x-axis points to the right, the positive y-axis points down, and the positive z-axis points forward.
		# output of motion data is consistent with convention that positive direction aligned with gravity leads to -1g and opposite direction leads to +1g, for example,
		# positive z_aixs points forward away from front glass of the device,
		#  1) if place the device flat on a table, facing up, positive z-axis points up, z-axis acceleration is around +1g
		#  2) facing down, positive z-axis points down, z-axis accleration would be around -1g
		#
		buckets_labels = ["Mounting screw pointing down, device facing out", "Mounting screw pointing left, device facing out", "Mounting screw pointing up, device facing out", "Mounting screw pointing right, device facing out", "Viewing direction facing down", "Viewing direction facing up"]

		gyro_bais = np.zeros(3, np.float32)
		old_settings = None
		if accel_file:
		    if gyro_file:
			#compute gyro bais

			#assume the first 4 seconds the device is still
			gyro = np.loadtxt(gyro_file, delimiter=",")
			gyro = gyro[gyro[:, 0] < gyro[0, 0]+4000, :]

			gyro_bais = np.mean(gyro[:, 1:], axis=0)
			print(gyro_bais)

		    #compute accel intrinsic parameters
		    max_norm = np.linalg.norm(np.array([0.5, 0.5, 0.5]))

		    measurements = [[], [], [], [], [], []]
		    import csv
		    with open(accel_file, 'r') as csvfile:
			reader = csv.reader(csvfile)
			rnum = 0
			for row in reader:
			    M = np.array([float(row[1]), float(row[2]), float(row[3])])
			    is_ok = False
			    for i in range(0, len(buckets)):
				if np.linalg.norm(M - buckets[i]) < max_norm:
				    is_ok = True
				    measurements[i].append(M)
			    rnum += 1
		    print('read %d rows.' % rnum)
		else:
		    print('Start interactive mode:')
		    if os.name == 'posix':
			old_settings = termios.tcgetattr(sys.stdin)
			tty.setcbreak(sys.stdin.fileno())

		    imu = imu_wrapper()
		    if not imu.enable_imu_device(serial_no):
			print('Failed to enable device.')
			return -1
		    measurements, gyro = imu.get_measurements(buckets, buckets_labels)
		    con_mm = np.concatenate(measurements)
		    if os.name == 'posix':
			termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

		    header = input('\nWould you like to save the raw data? Enter footer for saving files (accel_<footer>.txt and gyro_<footer>.txt)\nEnter nothing to not save raw data to disk. >')
		    print('\n')
		    if header:
			accel_file = 'accel_%s.txt' % header
			gyro_file = 'gyro_%s.txt' % header
			print('Writing files:\n%s\n%s' % (accel_file, gyro_file))
			np.savetxt(accel_file, con_mm, delimiter=',', fmt='%s')
			np.savetxt(gyro_file, gyro, delimiter=',', fmt='%s')
		    else:
			print('Not writing to files.')
		    # remove times from measurements:
		    measurements = [mm[:,1:] for mm in measurements]

		    gyro_bais = np.mean(gyro[:, 1:], axis=0)
		    print(gyro_bais)

		mlen = np.array([len(meas) for meas in measurements])
		print(mlen)
		print('using %d measurements.' % mlen.sum())

		nrows = mlen.sum()
		w = np.zeros([nrows, 4])
		Y = np.zeros([nrows, 3])
		row = 0
		for i in range(0, len(buckets)):
		    for m in measurements[i]:
			w[row, 0] = m[0]
			w[row, 1] = m[1]
			w[row, 2] = m[2]
			w[row, 3] = -1
			Y[row, 0] = buckets[i][0]
			Y[row, 1] = buckets[i][1]
			Y[row, 2] = buckets[i][2]
			row += 1
		np_version = [int(x) for x in np.version.version.split('.')]
		rcond_val = None if (np_version[1] >= 14 or np_version[0] > 1) else -1
		X, residuals, rank, singular = np.linalg.lstsq(w, Y, rcond=rcond_val)

		print(X)
		print("residuals:", residuals)
		print("rank:", rank)
		print("singular:", singular)
		check_X(X, w[:,:3], show_graph)

		calibration = {}

		if product_line == 'L500':
		    calibration["device_type"] = "L515"
		else:
		    calibration["device_type"] = "D435i"

		calibration["imus"] = list()
		calibration["imus"].append({})
		calibration["imus"][0]["accelerometer"] = {}
		calibration["imus"][0]["accelerometer"]["scale_and_alignment"] = X.flatten()[:9].tolist()
		calibration["imus"][0]["accelerometer"]["bias"] = X.flatten()[9:].tolist()
		calibration["imus"][0]["gyroscope"] = {}
		calibration["imus"][0]["gyroscope"]["scale_and_alignment"] = np.eye(3).flatten().tolist()
		calibration["imus"][0]["gyroscope"]["bias"] = gyro_bais.tolist()
		json_data = json.dumps(calibration, indent=4, sort_keys=True)

		directory = os.path.dirname(accel_file) if accel_file else '.'

		with open(os.path.join(directory,"calibration.json"), 'w') as outfile:
		    outfile.write(json_data)

		#concatinate the two 12 element arrays and save
		intrinsic_buffer = np.zeros([6,4])

		intrinsic_buffer[:3,:4] = X.T
		intrinsic_buffer[3:,:3] = np.eye(3)
		intrinsic_buffer[3:,3] = gyro_bais

		# intrinsic_buffer = ((np.array(range(24),np.float32)+1)/10).reshape([6,4])

		imu_calib_table = get_IMU_Calib_Table(intrinsic_buffer, product_line)

		with open(os.path.join(directory,"calibration.bin"), 'wb') as outfile:
		    outfile.write(imu_calib_table.astype('f').tostring())

		is_write = input('Would you like to write the results to the camera? (Y/N)')
		is_write = 'Y' in is_write.upper()
		if is_write:
		    print('Writing calibration to device.')

		    if product_line == 'L500':
			l500_send_command(dev, WRITE_TABLE, 0, 0, 0, 0, imu_calib_table)
		    else:
			calibration_table = get_calibration_table(imu_calib_table)
			eeprom = get_eeprom(calibration_table)
			write_eeprom_to_camera(eeprom, serial_no)

		    print('Done.')
		else:
		    print('Abort writing to device')

	    except Exception as e:
		print ('\nError: %s' % e)
	    finally:
		if os.name == 'posix' and old_settings is not None:
		    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

	    """
	    wtw = dot(transpose(w),w)
	    wtwi = np.linalg.inv(wtw)
	    print(wtwi)
	    X = dot(wtwi, Y)
	    print(X)
	    """
	if __name__ == '__main__':
	    main()
  ```
  
#### Realsense Distance camera calibration
  - Launch realsense-viwer and go to the tare calibration
![image](https://user-images.githubusercontent.com/90491438/165133988-fb7a6a3d-7ac3-406d-a8f4-68299dc65f8a.png)

#### RealSense extrinsic and intrinsic calibration
  - Launch realsense-viwer and go to the on-chip calibration
  - Example for extrinsic and intrinsic calibration: https://youtu.be/xiOlgkjn4kI

#### RealSense Exposure setting for calibration
- Launch realsense-viwer, off the depth image and on the infra image to view the 2 infra camera.
- Example for exposure setting calibration: https://youtu.be/V5VHD9SYvvU

### Recording the Rosbag file:
  - Make sure you off the emitter before you rosbag record the video. Also, change the frame rate to 6hz and camera height and width to (640 x 480).
  - Before recording, make sure to do all the required camera calibration for the realsense (imu calibration, distance calibration and camera calibration)
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


### Converting kalibr yaml file to jason file (Required file format for basalt VIO):
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


