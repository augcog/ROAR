#!/usr/bin/env python

from __future__ import division
import cv2
import numpy as np
import socket
import struct
import time

MAX_DGRAM = 2**15

def main():
    """ Getting image udp frame &
    concate before decode and output image """

    # Set up socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('192.168.1.10', 8002))
    dat = b''

    while True:
        data = s.recv(MAX_DGRAM)
        # seg, addr = s.recvfrom(MAX_DGRAM)
        # print(f"{addr} -> {seg}")
        count = int(data[0:3].decode('ascii'))
        print(f"{time.time()} -> {len(data)} -> {count}")
    s.close()


if __name__ == '__main__':
    main()