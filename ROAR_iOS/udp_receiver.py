import logging
import socket
import sys, os
from pathlib import Path
import time
from typing import Optional

sys.path.append(Path(os.getcwd()).parent.as_posix())
from ROAR.utilities_module.module import Module
from ROAR.utilities_module.utilities import get_ip
from collections import defaultdict

MAX_DGRAM = 9620


class UDPStreamer(Module):
    def save(self, **kwargs):
        pass

    def __init__(self, ios_address, port=8001, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(f"{self.name}")
        self.ios_addr = ios_address
        self.port = port
        self.s: Optional[socket.socket] = None
        self.start_socket()
        self.counter = 0
        self.logs = defaultdict(dict)

    def connect(self):
        pass

    def recv(self) -> Optional[bytes]:
        """
        If we have a really large data, we are going to chunk it and send it. Since iOS platform can only handle
        a maximum size of 9620, we use 9600 for each chunk here to make some room for extra header.

        So on the sender's side, the chunk size <= 9200 bytes of data + 3 + 3 + 3 bytes of header
        On the receiver's side, the chunk size <= 9200 bytes of data + 3 + 3 + 3 bytes of header

        And therefore, our buffer must be set to a number that is greater than or equal to 9209. We will take 9600

        ----------
        |  9600  |
        |--------|
        |  9600  |
        |--------|
        ...


        Each chunk is going to be structured like this:
        XXXYYYZZZDATA

        where XXX == 3 bytes long int encoded in ascii representing prefix num
        where YYY == 3 bytes long int encoded in ascii representing total num
        where ZZZ == 3 bytes long int encoded in ascii curr_buffer
        where DATA represent the actual data

        Returns:

        """
        buffer_num = -1
        log = dict()
        # this command might not be received.
        # If not received, it will cause s.recfrom to timeout, which will lead to restart of socket & return None
        # If received, server will attempt to send data over in chunk as specified above
        self.s.sendto(b'ack', (self.ios_addr, self.port))
        while True:
            try:
                seg, addr = self.s.recvfrom(MAX_DGRAM)  # this command might timeout
                prefix_num = int(seg[0:3].decode('ascii'))
                total_num = int(seg[3:6].decode('ascii'))
                curr_buffer = int(seg[6:9].decode('ascii'))
                if buffer_num == -1:
                    # initializing receiving sequence
                    buffer_num = curr_buffer
                    if prefix_num != 0:
                        # if the first one is not the starting byte, dump it.
                        buffer_num = -1
                        log = dict()
                    else:
                        # if the first one is the starting byte, start recording
                        log[prefix_num] = seg[9:]
                else:
                    if prefix_num in log:
                        # if i received a frame from another sequence
                        buffer_num = -1
                        log = dict()
                    else:
                        # if all checks passed, add it to the running recording of the data.
                        log[prefix_num] = seg[9:]

                if len(log) - 1 == total_num:
                    data = b''
                    for k in sorted(log.keys()):
                        data += log[k]
                    return data
            except socket.timeout as e:
                # if socket times out
                #   1. due to send ack not received, therefore server not sending data
                #   2. just lost packet causing not all packets are received
                self.restart_socket()
                return None

    def restart_socket(self):
        if self.s is not None:
            self.s.close()
        self.start_socket()

    def start_socket(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.settimeout(0.1)

    def _send_data(self, data: str):
        try:
            self.s.sendto(data.encode('utf-8'), (self.ios_addr, self.port))
            self.counter += 1
        except socket.timeout:
            print("timed out")
        except Exception as e:
            self.logger.error(e)

    def shutdown(self):
        super(UDPStreamer, self).shutdown()
        self.s.close()


if __name__ == '__main__':
    import numpy as np
    import cv2
    import struct

    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(name)s '
                               '- %(message)s',
                        datefmt="%H:%M:%S",
                        level=logging.DEBUG)
    udp_streamer = UDPStreamer(port=8001)
    udp_streamer.connect()
    while True:
        start = time.time()
        # print("about to recv")
        data = udp_streamer.recv()
        # d = np.frombuffer(data, dtype=np.float32)
        # print(d)
        # print(1 / (time.time() - start))

        # Receiving RGB
        img_data = data[16:]
        img = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

        if img is None:
            print("OH NO")

        if img is not None:
            try:
                cv2.imshow("img", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
                k = cv2.waitKey(1) & 0xff
                if k == ord('q') or k == 27:
                    break
                # print(f"{1 / (time.time() - start)} Image received")
            except Exception as e:
                print(e)

        """
        Receiving depth
        
        # intrinsics = data[0:32]
        # fx, fy, cx, cy = struct.unpack('f', intrinsics[0:4])[0], \
        #                  struct.unpack('f', intrinsics[8:12])[0], \
        #                  struct.unpack('f', intrinsics[16:20])[0], \
        #                  struct.unpack('f', intrinsics[24:28])[0]
        # intrinsics_array = np.array([
        #     [fx, 0, cx],
        #     [0, fy, cy],
        #     [0, 0, 1]
        # ])
        # # print(fx, fy, cx, cy)
        # img = np.frombuffer(data[32:], dtype=np.float32)
        # if img is None:
        #     continue
        # else:
        #     try:
        #         img = np.rot90(img.reshape((144, 256)), k=-1)
        #         cv2.imshow("img", img)
        #         cv2.waitKey(1)
        #         # print(f"{time.time()} Image received")
        #     except Exception as e:
        #         print(e)
        """
