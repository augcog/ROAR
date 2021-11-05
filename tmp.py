import logging
import time

from ROAR.utilities_module.utilities import get_ip
import qrcode
import cv2
import numpy as np
import socket
import struct

def showIPUntilAck():
    img = np.array(qrcode.make(f"{get_ip()}").convert('RGB'))
    success = False
    addr = None

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        s.bind((get_ip(), 8008))
        s.settimeout(1)
        while True:
            try:
                s.listen()

                cv2.imshow("Scan this code to connect to phone", img)
                k = cv2.waitKey(1) & 0xff
                if k == ord('q') or k == 27:
                    s.close()
                    break
                conn, addr = s.accept()
                addr = addr[0]
                if conn:
                    s.close()
                    success = True
                    break
            except socket.timeout as e:
                logging.info(f"Please tap on the ip address to scan QR code. ({get_ip()}:{8008}). {e}")
    except Exception as e:
        logging.error(f"Unable to bind socket: {e}")
    finally:
        s.close()
        cv2.destroyWindow("Scan this code to connect to phone")
    return success, addr


def recv(s: socket.socket):
    buffer_num = -1
    log = dict()
    while True:
        seg, addr = s.recvfrom(9620)
        prefix_num = int(seg[0:3].decode('ascii'))
        total_num = int(seg[3:6].decode('ascii'))
        curr_buffer = int(seg[6:9].decode('ascii'))

        # print(f"BEFORE curr_buff = {curr_buffer} | prefix_num = {prefix_num} "
        #       f"| total_num = {total_num} | len(log) = {len(log)}")
        if buffer_num == -1:
            # initializing
            buffer_num = curr_buffer
            if prefix_num != 0:
                # if the first one is not the starting byte, dump it.
                print("the first one is not the starting byte")
                buffer_num = -1
                log = dict()
            else:
                # if the first one is the starting byte, start recording
                log[prefix_num] = seg[9:]
        else:
            if prefix_num in log:
                # if i received a frame from another sequence
                print("i received a frame from another sequence")
                buffer_num = -1
                log = dict()
            else:
                log[prefix_num] = seg[9:]
        # print(f"AFTER curr_buff = {curr_buffer} | prefix_num = {prefix_num} | total_num = {total_num} "
        #       f"| len(log) = {len(log)} | log.keys = {list(sorted(log.keys()))}")

        if len(log) - 1 == total_num:
            data = b''
            for k in sorted(log.keys()):
                data += log[k]
            # print()
            return data


def client_test():
    msgFromClient = "Hello UDP Server"
    bytesToSend = str.encode(msgFromClient)
    serverAddressPort = ("10.0.0.26", 8001)
    bufferSize = 1024
    # Create a UDP socket at client side
    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    while True:
        # Send to server using created UDP socket
        start = time.time()
        UDPClientSocket.sendto(b"hello", serverAddressPort)

        # print("message sent")
        data = recv(UDPClientSocket)

        img_data = data[16:]
        intrinsics = data[:16]
        fx, fy, cx, cy = struct.unpack('f', intrinsics[0:4])[0], \
                         struct.unpack('f', intrinsics[4:8])[0], \
                         struct.unpack('f', intrinsics[8:12])[0], \
                         struct.unpack('f', intrinsics[12:16])[0]
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        # print(intrinsics)
        img = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
        cv2.imshow("img", img)
        cv2.waitKey(1)
        print(f"fps = {1 / (time.time() - start)}")


if __name__ == '__main__':
    client_test()
