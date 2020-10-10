import zmq
import pickle
import numpy as np
import struct
import time

DETECT_PORT = '12345'

context = zmq.Context()
socket = context.socket(zmq.SUB)

## SUB side better to use connect, it will get lastest data
socket.connect("tcp://127.0.0.1:" + DETECT_PORT)
#socket.bind("tcp://127.0.0.1:"+port)
socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))


## For non-blocking better to use poller
poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)
timeout = 1 # ms

while True:

	## In case of non-blocking process
	try:
		socks = dict(poller.poll(timeout))
	except KeyboardInterrupt:
		break

	if socket in socks:
		data = socket.recv(zmq.NOBLOCK)
		parse_data = pickle.loads(data)
		print(parse_data)
		
	time.sleep(0.1)
