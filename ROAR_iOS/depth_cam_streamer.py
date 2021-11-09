from typing import List, Optional, Tuple, List
import numpy as np
import cv2
import sys, os
from pathlib import Path

sys.path.append(Path(os.getcwd()).parent.as_posix())
from ROAR_iOS.udp_receiver import UDPStreamer
import struct
import time

MAX_DGRAM = 9600


class DepthCamStreamer(UDPStreamer):
    def __init__(self, resize: Optional[Tuple] = None, **kwargs):
        super().__init__(**kwargs)
        self.curr_image: Optional[np.ndarray] = None
        self.resize = resize
        self.intrinsics: Optional[np.ndarray] = None

    def run_in_series(self, **kwargs):
        try:
            data = self.recv()
            if data is None:
                return
            img_data = data[16:]
            intrinsics = data[0:16]
            fx, fy, cx, cy = struct.unpack('f', intrinsics[0:4])[0], \
                             struct.unpack('f', intrinsics[4:8])[0], \
                             struct.unpack('f', intrinsics[8:12])[0], \
                             struct.unpack('f', intrinsics[12:16])[0]
            self.intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            img = np.frombuffer(img_data, dtype=np.float32)
            if img is not None:
                self.curr_image = np.rot90(img.reshape((144, 256)), k=-1)

        except OSError:
            self.should_continue_threaded = False
        except Exception as e:
            self.logger.error(e)


if __name__ == '__main__':
    ir_image_server = DepthCamStreamer(ios_address="10.0.0.26",
                                       port=8002,
                                       name="world_depth_streamer",
                                       update_interval=0.05,
                                       threaded=True)
    # ir_image_server.connect()
    while True:
        ir_image_server.run_in_series()
        if ir_image_server.curr_image is not None:
            img = ir_image_server.curr_image
            cv2.imshow("img", img)
            cv2.waitKey(1)
