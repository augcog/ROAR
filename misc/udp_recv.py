#!/usr/bin/env python

from __future__ import division
import cv2
import numpy as np
import socket
import struct

MAX_DGRAM = 9600


def dump_buffer(s):
    """ Emptying buffer frame """
    while True:
        seg, addr = s.recvfrom(MAX_DGRAM)
        prefix_num = int(seg[0:3].decode('ascii'))
        if prefix_num == 1:
            print("finish emptying buffer")
            break


def main():
    """ Getting image udp frame &
    concate before decode and output image """

    # Set up socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('10.105.209.67', 8001))
    dat = b''
    dump_buffer(s)

    while True:
        seg, addr = s.recvfrom(MAX_DGRAM)
        prefix_num = int(seg[0:3].decode('ascii'))
        if prefix_num > 1:
            dat += seg[3:]
        else:
            dat += seg[3:]
            try:
                img = np.frombuffer(dat, dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                cv2.imshow('recv', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(e)
            dat = b''

    cv2.destroyAllWindows()
    s.close()


if __name__ == "__main__":
    main()
