import darknet as dn
import cv2
import pyrealsense2 as rs
import numpy as np
import time
import socket
import struct
import pickle
import zmq
import base64

########################## SWITCH FLAG ########################## 
STREAM_SHOW = True			# stream a video to our laptop
LOCAL_SHOW = False			# show on local display
AUTOPILOT = True			# send bbox_data to another script

REALSENSE_CAM = False		# using Intel Realsense camera
WEBCAM = not REALSENSE_CAM	# or using normal webcam


########################## DARKNET INITIALIZATION ########################## 
net = dn.load_net(b"darknet/chestnut/cfg/yolov3-tiny-obj_small_test.cfg", b"darknet/chestnut/backup/yolov3-tiny-obj_small_55000.weights", 0)
meta = dn.load_meta(b"darknet/chestnut/data/obj.data")

########################## VIDEO STREAMER ########################## 
PORT_DISPLAY = '5555'
context = zmq.Context()
if STREAM_SHOW:
    
    footage_socket = context.socket(zmq.PUB)
    footage_socket.bind('tcp://*:' + PORT_DISPLAY)

    print("Video streaming on PORT: " +PORT_DISPLAY)

########################## UDP ########################## 
## For some reason, detection script doesn't receive ROBOT_STATUS message
## when using ZMQ. But general UDP socket is working fine

# RECEIVER_IP = "127.0.0.1"
# DETECT_PORT = '12345'
# detect_sock = context.socket(zmq.PUB)
# detect_sock.bind("tcp://*:" + DETECT_PORT)

# ROBOT_STATUS_PORT = '6666'
# robot_sock = context.socket(zmq.SUB)
# robot_sock.connect("tcp://127.0.0.1:" + ROBOT_STATUS_PORT)

# ## For non-blocking better to use poller
# poller = zmq.Poller()
# poller.register(robot_sock, zmq.POLLIN)
# timeout = 1 # ms

DETECT_PORT = 12345
detect_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

ROBOT_STATUS_PORT = 22334
robot_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
robot_sock.bind(("0.0.0.0", ROBOT_STATUS_PORT))
robot_sock.setblocking(0)


if REALSENSE_CAM:
	bbox_data = {
		'nboxes': 0,
		'depth': np.zeros(16,dtype=float),
		'center': np.zeros((16,2), dtype=float),
		'pos_x' : np.zeros(16,dtype=float)
	}
else:
	bbox_data = {
		'nboxes': 0,
		'box_height': np.zeros(16,dtype=float),
		'xc': np.zeros(16,dtype=float),
		'yc': np.zeros(16,dtype=float)
	}


########################## CAMERA ##########################
frame_width = 640		# 848    1280   1920
frame_height = 480		# 480	 720	1080

if REALSENSE_CAM:
	pipeline = rs.pipeline()
	config = rs.config()
	#config.enable_device('909512072318')  # specific cam
	config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, 30)
	pipeline.start(config)

	profile = pipeline.get_active_profile()
	depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
	depth_intrinsics = depth_profile.get_intrinsics()
else:
	video = cv2.VideoCapture(0)
	video.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
	video.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

########################## LOOP ########################## 
fps = 0.0
fps2 = 0.0
tic = time.time()
tic2 = time.time()
k = 0

robot_status = None
t1 = time.time()
t2 = time.time()

try:
	while True:

		# ## In case of non-blocking process
		# try:
		# 	socks = dict(poller.poll(timeout))
		# except KeyboardInterrupt:
		# 	break

		# if robot_sock in socks:
		# 	data = robot_sock.recv(zmq.NOBLOCK)
		# 	parse_data = pickle.loads(data)
		# 	robot_status = parse_data
		# 	print("receive status")
		# 	print(robot_status)

		## get ROBOT status, probably READY or BUSY
		try:
			data, addr = robot_sock.recvfrom(1024)
			parse_data = pickle.loads(data)
			t1 = time.time()
			if (abs(t2 - t1) > 0.2):
				t2 = time.time()
				print('robot_status: ', parse_data)
		except socket.error:
			pass
		else:
			robot_status = parse_data

		######################## Get frame from camera #############################
		if REALSENSE_CAM:
			frames = pipeline.wait_for_frames()
			color_frame = frames.get_color_frame()
			depth_frame = frames.get_depth_frame()

			depth_image = np.asanyarray(depth_frame.get_data())
			color_image = np.asanyarray(color_frame.get_data())

			## Scanline ##
			scanLine = np.array([])
			for i in range(frame_width):
				scanLine = np.append(scanLine, depth_frame.get_distance(i,int(frame_height/2)))

		else:
			ret, color_image = video.read()


		## Copy original image frame ##
		if LOCAL_SHOW:
			show_frame = color_image
		if STREAM_SHOW:
			stream_frame = color_image


		## Robot is BUSY (picking nuts), DON'T do detection
		if robot_status == 'BUSY':
			#print("Stop detection")
			r = None
		else:
			r = dn.detect(net, meta, color_image, thresh=0.25, hier_thresh=0.5, nms=0.45)

		
		######################## Detected Something ##############################
		if r != None:
			if len(r) > 0:

				print("nboxes: "+"{:d}".format(len(r)))

				for i in range(len(r)):
					className = str(r[i][0])        # Convert byte value to string
					className = className[2:len(className)-1]   # This is to remove b'....' on name
					#print("className",className)

					if className == 'chestnut':
						xmin = int(r[i][2][0]-r[i][2][2]/2)
						ymin = int(r[i][2][1]-r[i][2][3]/2)
						xmax = int(r[i][2][0]+r[i][2][2]/2)
						ymax = int(r[i][2][1]+r[i][2][3]/2)

						xc = (xmax+xmin)/2
						yc = (ymax+ymin)/2

						if REALSENSE_CAM:
							Depth = depth_frame.get_distance(int(xc),int(yc))
							bbox_data['center'][k][0] = xc
							bbox_data['center'][k][1] = yc
							points=rs.rs2_deproject_pixel_to_point(depth_intrinsics,[int(xc),int(yc)],Depth)
							bbox_data['depth'][k] = Depth
							bbox_data['pos_x'][k] = points[0]
							bbox_data['nboxes'] = k+1
						else:

							if ymin < 0:
								ymin = 0
							if ymax > frame_height:
								ymax = int(frame_height)

							
							if xc == 0.0 and yc == 0.0:
								pass
							else:
								box_height = ymax-ymin
								bbox_data['box_height'][k] = box_height
								bbox_data['xc'][k] = xc
								bbox_data['yc'][k] = yc
								bbox_data['nboxes'] = k+1

								# Just keep printing on screen, to see it's not stuck somewhere
								#print("xc: {:.6f}".format(bbox_data['xc'][k]) + " yc: {:.6f}".format(bbox_data['yc'][k]))


						if LOCAL_SHOW:
							cv2.rectangle(show_frame, (xmin, ymin), (xmax, ymax), (0,0,255), 10)

						if STREAM_SHOW:
							cv2.rectangle(stream_frame, (xmin, ymin), (xmax, ymax), (0,0,255), 10)
							cv2.putText(stream_frame, "chestnut", (xmin, ymin-10), 1, 2, (0,0,255), 1, cv2.LINE_AA)
							

						k += 1

					

				k = 0

				
		######################## Detected Nothing ##############################
		## we need to clear data out otherwise, it will send old data instead
		else:
			# Clear data
			if REALSENSE_CAM:
				bbox_data['nboxes'] = 0
				bbox_data['depth'] = np.zeros(16,dtype=float)
				bbox_data['pos_x'] = np.zeros(16,dtype=float)
				bbox_data['center'] = np.zeros((16, 2), dtype=float)
			else:
				bbox_data['nboxes'] = 0
				bbox_data['box_height'] =  np.zeros(16,dtype=float)
				bbox_data['xc'] =  np.zeros(16,dtype=float)
				bbox_data['yc'] =  np.zeros(16,dtype=float)


		if AUTOPILOT:
			# UDP send
			detection_packet = pickle.dumps(bbox_data, zmq.NOBLOCK)
			detect_sock.sendto(detection_packet,("127.0.0.1",DETECT_PORT))
			#detect_sock.send(detection_packet)

			if REALSENSE_CAM:
				scanLine_packet = pickle.dumps(scanLine)
				scan_sock.sendto(scanLine_packet, (RECEIVER_IP, SCAN_PORT))


		if LOCAL_SHOW:
			
			#depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)
			
			cv2.putText(show_frame, "fps: {:.2f}".format(fps), (10,50), 1, 2, (255,0,0), 2, cv2.LINE_AA)
			cv2.imshow("Local Show", show_frame)
			toc = time.time()
			curr_fps = 1.0 / (toc - tic)
			# calculate an exponentially decaying average of fps number
			fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
			tic = toc

			key = cv2.waitKey(1)
			if key == 27:  # ESC key: quit program
				break


		if STREAM_SHOW:
			#grabbed, frame = camera.read()  # grab the current frame
			#frame = cv2.resize(frame, (640, 480))  # resize the frame
			
			cv2.putText(stream_frame, "fps: {:.2f}".format(fps2), (10,50), 1, 2, (255,0,0), 2, cv2.LINE_AA)

			encoded, buffer = cv2.imencode('.jpg', stream_frame)
			jpg_as_text = base64.b64encode(buffer)
			footage_socket.send(jpg_as_text)
			toc2 = time.time()
			curr_fps2 = 1.0 / (toc2 - tic2)
			# calculate an exponentially decaying average of fps number
			fps2 = curr_fps2 if fps2 == 0.0 else (fps2*0.95 + curr_fps2*0.05)
			tic2 = toc2

		

		## keep printing if found out something / when ROBOT is BUSY, it won't do detection and print 0
		# print("nboxes: "+"{:d}".format(bbox_data['nboxes']))
		


finally:
	cv2.destroyAllWindows()
	# Stop streaming
	if REALSENSE_CAM:
		pipeline.stop()