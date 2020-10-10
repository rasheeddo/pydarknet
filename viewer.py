import cv2
import zmq
import base64
import numpy as np
import socket
import struct

# Use to stream video
context = zmq.Context()
footage_socket = context.socket(zmq.SUB)

footage_socket.connect('tcp://192.168.8.161:5555')
footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))


while True:

    try:
        frame = footage_socket.recv_string()
        img = base64.b64decode(frame)
        npimg = np.fromstring(img, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)

        #source = cv2.resize(source, (1280,960))

        cv2.imshow("Stream", source)
        cv2.waitKey(1)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        break
