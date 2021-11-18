import socket
from ROAR.utilities_module.utilities import get_ip

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((get_ip(), 8008))
s.send(b'data')
print("sent")