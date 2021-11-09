import cv2
import numpy as np
import socket

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 12345
    while True:
        s.sendto(b'hello world', ("192.168.1.10", 8001))
