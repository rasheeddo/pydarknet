import zmq
import pickle
import numpy as np
import struct
import time
import socket

### Using ZMQ socket
# DETECT_PORT = '12345'
# context = zmq.Context()
# socket = context.socket(zmq.SUB)
# ## SUB side better to use connect, it will get lastest data
# socket.connect("tcp://127.0.0.1:" + DETECT_PORT)
# #socket.bind("tcp://127.0.0.1:"+port)
# socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
# ## For non-blocking better to use poller
# poller = zmq.Poller()
# poller.register(socket, zmq.POLLIN)
# timeout = 1 # ms

### Using regular python socket
DETECT_PORT = 12345
detect_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
detect_sock.bind(("0.0.0.0", DETECT_PORT))
detect_sock.setblocking(0)

while True:

	### Using ZMQ socket
	# ## In case of non-blocking process
	# try:
	# 	socks = dict(poller.poll(timeout))
	# except KeyboardInterrupt:
	# 	break

	# if socket in socks:
	# 	data = socket.recv(zmq.NOBLOCK)
	# 	parse_data = pickle.loads(data)
	# 	print(parse_data)
		
	# time.sleep(0.1)


	### Using regular python socket
	try:
		data, addr = detect_sock.recvfrom(30000)
		parse_data = pickle.loads(data)
	except socket.error:
		pass
	else:
		print(parse_data)
