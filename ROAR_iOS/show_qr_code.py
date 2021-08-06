import time
import socket
import qrcode, numpy
import cv2


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


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