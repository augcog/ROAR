import time
import socket
import qrcode, numpy
import cv2
import sys
import os
from pathlib import Path
curr_path = Path(os.getcwd())
sys.path.append(curr_path.parent.as_posix())
from ROAR.utilities_module.utilities import get_ip


HOST = get_ip()
PORT = 8006
cv2.imshow("qr code", numpy.array(qrcode.make(f"{get_ip()}").convert('RGB')))
cv2.waitKey(1)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

s.bind((HOST, PORT))
s.listen()
conn, addr = s.accept()
cv2.destroyWindow("qr code")
conn.close()
s.close()
with open("./data/iphone_ip.txt", 'w') as f:
    f.write(addr[0] + "\n")
print(f"Iphone IP = ", addr[0])

