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
from goprocam import GoProCamera, constants
import argparse

parser = argparse.ArgumentParser(description='human detection from pydarknet yolov3-tiny')
parser.add_argument('--camera', help="camera device e.g. webcam, realsense, gopro")
parser.add_argument('--udpout', type=int, help="set to 0 or 1 to stream data out at specified UDP port")
parser.add_argument('--show', help="show an image on local pc or stream out at port 5555 by zmq")
# parser.add_argument('--baudrate', type=float,
#                     help="Vehicle connection baudrate. If not specified, a default value will be used.")
# parser.add_argument('--obstacle_distance_msg_hz', type=float,
#                     help="Update frequency for OBSTACLE_DISTANCE message. If not specified, a default value will be used.")
# parser.add_argument('--debug_enable',type=float, help="Enable debugging information")
args = parser.parse_args()
CAM = args.camera
UDPOUT = args.udpout
SHOW = args.show
# FCU connection variables

if CAM is not None:
	if CAM == "webcam":
		WEBCAM = True
		REALSENSE_CAM = False
		GOPRO = False
	elif CAM == "realsense":
		WEBCAM = False
		REALSENSE_CAM = True
		GOPRO = False
	elif CAM == "gopro":
		WEBCAM = False
		REALSENSE_CAM = False
		GOPRO = True
	else:
		print("Error, camera is not one of webcam/realsense/gopro")
		quit()
else:
	parser.print_help()
	quit()

if (UDPOUT is None):
	AUTOPILOT = False
elif (UDPOUT > 1024) and (UDPOUT < 65535):
	AUTOPILOT = True
	DETECT_PORT = int(UDPOUT)
else:
	print("Error, invalid udpout")
	parser.print_help()
	quit()

if SHOW is not None:
	if SHOW == 'local':
		LOCAL_SHOW = True
		STREAM_SHOW = False
	elif SHOW == 'stream':
		if GOPRO:
			LOCAL_SHOW = True
			STREAM_SHOW = False
			print("I don't suggest to stream gopro out to the viewer with pydarknet, but local show is fine")
		else:
			STREAM_SHOW = True
			LOCAL_SHOW = False
	else:
		STREAM_SHOW = False
		LOCAL_SHOW = False
else:
	STREAM_SHOW = False
	LOCAL_SHOW = False


########################## SWITCH FLAG ########################## 
# STREAM_SHOW = False			# stream a video to our laptop
# LOCAL_SHOW = True			# show on local display
# AUTOPILOT = True			# send bbox_data to another script

# REALSENSE_CAM = False		# using Intel Realsense camera
# WEBCAM = False				# or using normal webcam
# GOPRO = True


########################## DARKNET INITIALIZATION ########################## 
net = dn.load_net(b"darknet/human/cfg/yolov3-tiny.cfg", b"darknet/human/backup/yolov3-tiny.weights", 0)
meta = dn.load_meta(b"darknet/human/cfg/coco.data")

########################## VIDEO STREAMER ########################## 
PORT_DISPLAY = '5555'

if STREAM_SHOW:
	context = zmq.Context()
	footage_socket = context.socket(zmq.PUB)
	footage_socket.bind('tcp://*:' + PORT_DISPLAY)

	print("Video streaming on PORT: " +PORT_DISPLAY)

########################## UDP ########################## 
RECEIVER_IP = "127.0.0.1"
# DETECT_PORT = 12345
SCAN_PORT = 3100
detect_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
scan_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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
		'center': np.zeros((16,2),dtype=float),
	}


########################## CAMERA ##########################
frame_width = 848		# 848    1280   1920
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

	depth_sensor = profile.get_device().first_depth_sensor()
	depth_scale = depth_sensor.get_depth_scale()

	align_to = rs.stream.color
	align = rs.align(align_to)

elif WEBCAM:
	video = cv2.VideoCapture(0)
	video.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
	video.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

elif GOPRO:
	## This ip_address seems to be constant when using USB Connected mode
	gopro = GoProCamera.GoPro(ip_address="172.26.174.51")
	gopro.webcamFOV(constants.Webcam.FOV.Wide)
	## we use gopro as webcam with slightly low resolution, or can try with 1080 as well, but 1080 will be a bit slow
	gopro.startWebcam(resolution="480")   #480
	# video = cv2.VideoCapture("udp://172.26.174.51:8554?overrun_nonfatal=1&fifo_size=50000000", cv2.CAP_FFMPEG)   # , cv2.CAP_FFMPEG
	video = cv2.VideoCapture("udp://172.26.174.51:8554", cv2.CAP_FFMPEG)

########################## LOOP ########################## 
fps = 0.0
fps2 = 0.0
tic = time.time()
tic2 = time.time()
k = 0

try:
	while True:

		######################## Get frame from camera #############################
		if REALSENSE_CAM:
			frames = pipeline.wait_for_frames()
			# Align the depth frame to color frame
			aligned_frames = align.process(frames)

			color_frame = frames.get_color_frame()
			#depth_frame = frames.get_depth_frame()
			aligned_depth_frame = aligned_frames.get_depth_frame()

			#depth_image = np.asanyarray(depth_frame.get_data())
			depth_image = np.asanyarray(aligned_depth_frame.get_data())
			color_image = np.asanyarray(color_frame.get_data())
			depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)
			# Scanline ##
			scanLine = np.array([])
			for i in range(frame_width):
				scanLine = np.append(scanLine, aligned_depth_frame.get_distance(i,int(frame_height/2)))

		else:
			ret, color_image = video.read()


		# Copy original image frame ##
		if LOCAL_SHOW:
			if REALSENSE_CAM:
				# print(color_image.shape)
				# print(depth_colormap.shape)
				show_frame = np.hstack((color_image, depth_colormap))
			else:
				show_frame = color_image
				# print(show_frame.shape)
		if STREAM_SHOW:
			stream_frame = color_image


		## Detection ##
		r = dn.detect(net, meta, color_image, thresh=0.5, hier_thresh=0.5, nms=0.45)

		######################## Detected Something ##############################
		if len(r) > 0:

			for i in range(len(r)):
				className = str(r[i][0])        # Convert byte value to string
				className = className[2:len(className)-1]   # This is to remove b'....' on name
				#print("className",className)

				if className == 'person':
					xmin = int(r[i][2][0]-r[i][2][2]/2)
					ymin = int(r[i][2][1]-r[i][2][3]/2)
					xmax = int(r[i][2][0]+r[i][2][2]/2)
					ymax = int(r[i][2][1]+r[i][2][3]/2)

					xc = (xmax+xmin)/2
					yc = (ymax+ymin)/2

					if REALSENSE_CAM:
						
						Depth = aligned_depth_frame.get_distance(int(xc),int(yc))
						bbox_data['center'][k][0] = xc
						bbox_data['center'][k][1] = yc
						#points=rs.rs2_deproject_pixel_to_point(depth_intrinsics,[int(xc),int(yc)],Depth)
						bbox_data['depth'][k] = Depth
						#bbox_data['pos_x'][k] = points[0]
						
						# depth = depth_image[xmin:xmax,ymin:ymax].astype(float)
						# depth = depth * depth_scale
						# dist,_,_,_ = cv2.mean(depth)
						# print(dist)
						
						bbox_data['nboxes'] = k+1
					else:

						if ymin < 0:
							ymin = 0
						if ymax > frame_height:
							ymax = int(frame_height)

						box_height = ymax-ymin
						bbox_data['box_height'][k] = box_height
						bbox_data['center'][k][0] = xc
						bbox_data['center'][k][1] = yc
						bbox_data['nboxes'] = k+1


					if LOCAL_SHOW:
						if WEBCAM or GOPRO:
							cv2.rectangle(show_frame, (xmin, ymin), (xmax, ymax), (0,0,255), 10)
						else:
							color_image = cv2.circle(color_image, (int(xc),int(yc)), 10, (255,255,255), 3)
							depth_colormap = cv2.circle(depth_colormap, (int(xc),int(yc)), 10, (255,255,255), 3)
							cv2.rectangle(color_image, (xmin, ymin), (xmax, ymax), (0,0,255), 10)
							cv2.rectangle(depth_colormap, (xmin, ymin), (xmax, ymax), (0,0,255), 10)

					if STREAM_SHOW:
						cv2.rectangle(stream_frame, (xmin, ymin), (xmax, ymax), (0,0,255), 10)
						cv2.putText(stream_frame, "height: {:.2f}".format(box_height), (10,80), 1, 2, (0,255,0), 2, cv2.LINE_AA)
						cv2.putText(stream_frame, "center: {:.2f}".format(xc), (10,110), 1, 2, (0,125,125), 2, cv2.LINE_AA)

					k += 1

			k = 0


		######################## Detected Nothing ##############################
		else:
			# Clear data
			if REALSENSE_CAM:
				bbox_data['nboxes'] = 0
				bbox_data['depth'] = np.zeros(16,dtype=float)
				bbox_data['pos_x'] = np.zeros(16,dtype=float)
				bbox_data['center'] = np.zeros((16, 2), dtype=float)
			else:
				bbox_data['nboxes'] = 0
				bbox_data['box_height'] = np.zeros(16,dtype=float)
				bbox_data['center'] = np.zeros((16,2),dtype=float)


		if AUTOPILOT:
			# UDP send
			detection_packet = pickle.dumps(bbox_data)
			detect_sock.sendto(detection_packet,(RECEIVER_IP,DETECT_PORT))

			if REALSENSE_CAM:
				scanLine_packet = pickle.dumps(scanLine)
				scan_sock.sendto(scanLine_packet, (RECEIVER_IP, SCAN_PORT))


		if LOCAL_SHOW:

			if REALSENSE_CAM:
				# depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)
				show_frame = np.hstack((color_image, depth_colormap))

			if (WEBCAM or GOPRO) and frame_height < 960:
				show_frame = cv2.resize(show_frame, (1696, 960))

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


		# Just keep printing on screen, to see it's not stuck somewhere
		# print("nboxes: "+"{:d}".format(bbox_data['nboxes']))


finally:
	cv2.destroyAllWindows()
	# Stop streaming
	if REALSENSE_CAM:
		pipeline.stop()