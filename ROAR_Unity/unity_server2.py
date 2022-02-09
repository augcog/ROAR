import socket
import logging
import threading
from ROAR.utilities_module.vehicle_models import VehicleControl
from ROAR.utilities_module.module import Module
from typing import Optional, Tuple, Any, List
from abc import ABC, abstractmethod
import numpy as np
import math
import cv2
import struct


class UnityServer(Module, ABC):
    def recv(self) -> Optional[Tuple[bytes, Any]]:
        try:
            bytes_addr = self.s.recvfrom(self.buffer_size)
            message = bytes_addr[0]
            address = bytes_addr[1]
            return message, address
        except Exception as e:
            # self.logger.error(e)
            return None

    def __init__(self, host: str, port: int, buffer_size: int = 9200, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(self.name)
        self.host = host
        self.port = port
        self.buffer_size = buffer_size

        self.s = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.__init_socket()

    def __init_socket(self):
        self.s.bind((self.host, self.port))
        self.s.settimeout(0.1)

    def send_data(self, data: bytes, address):
        upload_chunk_size = 9200
        total_size = len(data)
        offset = 0
        counter = 0
        total = int(math.ceil(total_size / upload_chunk_size)) - 1

        while offset < total_size:
            data_to_send: bytes = f"{counter}".rjust(3, "0").encode('ascii')
            data_to_send += (f"{total}".rjust(3, "0").encode('ascii'))
            data_to_send += (f"{0}".rjust(3, "0").encode('ascii'))
            self.logger.debug(f"{counter}, {total}, {len(data)}")
            chunk_size = total_size - offset if offset + upload_chunk_size < self.buffer_size else upload_chunk_size
            print(f"offset = {offset} | chunk_size = {chunk_size} | total_size = {total_size}")
            chunk = data[offset:offset + chunk_size]
            data_to_send += chunk
            self.s.sendto(data_to_send, address)
            offset += chunk_size
            counter += 1


class UnityRGBServer(UnityServer):
    def __init__(self, host: str, port: int, **kwargs):
        super().__init__(host, port, **kwargs)
        self.data_to_send: Optional[bytes] = None
        self.view_offset = 200

    def run_in_series(self, **kwargs):
        d = self.recv()
        if d is not None and self.data_to_send is not None:
            self.send_data(self.data_to_send, d[1])

    def save(self, **kwargs):
        pass

    def update_image(self, new_image: np.ndarray, intrinsics: np.ndarray):
        fx, fy, cx, cy = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]
        s = new_image.shape
        height = s[1] // 4
        self.view_offset = np.clip(self.view_offset, 0, s[1] - height)
        min_y = s[1] - height - self.view_offset
        max_y = s[1] - self.view_offset
        new_image = cv2.flip(cv2.rotate(new_image, cv2.ROTATE_90_COUNTERCLOCKWISE)[min_y: max_y, :], 1)

        cv2.imshow("new_img", new_image)
        cv2.waitKey(1)

        is_success, im_buf_arr = cv2.imencode(".jpg", new_image)
        img_buf = im_buf_arr.tobytes()
        self.data_to_send = struct.pack('f', fx) + struct.pack('f', fy) + \
                            struct.pack('f', cx) + struct.pack('f', cy)
        self.data_to_send += img_buf


class UnityVehicleStateServer(UnityServer):
    def __init__(self, host: str, port: int, **kwargs):
        super().__init__(host, port, **kwargs)
        self.data_to_send: Optional[bytes] = None

    def run_in_series(self, **kwargs):
        d = self.recv()
        if d is not None and self.data_to_send is not None:
            self.send_data(self.data_to_send, d[1])

    def save(self, **kwargs):
        pass

    def update_state(self, x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, gx, gy, gz, recv_time):
        vals = [x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, gx, gy, gz, recv_time]
        data_to_send = ",".join([str(v) for v in vals])
        self.data_to_send = data_to_send.encode('utf-8')


class UnityControlServer(UnityServer):
    def __init__(self, host: str, port: int, **kwargs):
        super().__init__(host, port, **kwargs)
        self.vehicle_control: VehicleControl = VehicleControl()

    def run_in_series(self, **kwargs):
        d = self.recv()
        if d is not None:
            data: List[str] = d[0].decode('utf-8').split(',')
            self.vehicle_control.throttle = float(data[0])
            self.vehicle_control.steering = float(data[1])

    def save(self, **kwargs):
        pass
