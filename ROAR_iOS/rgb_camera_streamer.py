import time
from typing import List, Optional, Tuple, List
import cv2
import numpy as np
import sys, os
from pathlib import Path
import time

sys.path.append(Path(os.getcwd()).parent.as_posix())
from ROAR_iOS.udp_receiver import UDPStreamer
import struct
from collections import deque

MAX_DGRAM = 9600


class RGBCamStreamer(UDPStreamer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.curr_image: Optional[np.ndarray] = None
        self.intrinsics: Optional[np.ndarray] = None

    def run_in_series(self, **kwargs):
        try:
            data = self.recv()
            if data is None:
                return
            img_data = data[16:]
            intrinsics = data[:16]
            fx, fy, cx, cy = struct.unpack('f', intrinsics[0:4])[0], \
                             struct.unpack('f', intrinsics[4:8])[0], \
                             struct.unpack('f', intrinsics[8:12])[0], \
                             struct.unpack('f', intrinsics[12:16])[0]
            self.intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])

            img = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
            if img is not None:
                self.curr_image = img

        except OSError:
            self.should_continue_threaded = False

        except Exception as e:
            self.logger.error(e)


if __name__ == '__main__':
    ir_image_server = RGBCamStreamer(ios_address="10.0.0.26",
                                     pc_port=8001,
                                     name="world_rgb_streamer",
                                     update_interval=0.025,
                                     threaded=True)
    # ir_image_server.connect()
    while True:
        ir_image_server.run_in_series()
        if ir_image_server.curr_image is not None:
            img = ir_image_server.curr_image
            cv2.imshow("img", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
            cv2.waitKey(1)
    # ir_image_server.run_in_series()
