import darknet as dn
import cv2

video = cv2.VideoCapture('/dev/video0')

net = dn.load_net(b"darknet/cfg/yolov3-tiny.cfg", b"darknet/backup/yolov3-tiny.weights", 0)
meta = dn.load_meta(b"darknet/cfg/coco.data")

while True:

	ret, frame = video.read()

	r = dn.detect(net, meta, frame, thresh=0.5, hier_thresh=0.5, nms=0.45)

	print(r)

	
