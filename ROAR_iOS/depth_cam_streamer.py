import logging
from websocket import create_connection
from typing import List, Optional, Tuple, List
import cv2
import numpy as np
from pathlib import Path
from ROAR.utilities_module.module import Module

import datetime


class DepthCamStreamer(Module):
    def save(self, **kwargs):
        # no need to save. use Agent's saving mechanism
        pass

    def __init__(self, host, port, show=False, resize: Optional[Tuple] = None,
                 name: str = "depth_cam", threaded: bool = True,
                 update_interval: float = 0.5):
        super().__init__(threaded=threaded, name=name, update_interval=update_interval)
        self.logger = logging.getLogger(f"{self.name} server on [{host}:{port}]")
        self.host = host
        self.port = port
        self.ws = None

        self.resize = resize
        self.show = show

        self.curr_image: Optional[np.ndarray] = None
        self.intrinsics: Optional[np.ndarray] = None
        self.logger.info(f"{name} initialized")

    def receive(self):
        try:
            self.ws = create_connection(f"ws://{self.host}:{self.port}/{self.name}", timeout=0.1)
            im = self.ws.recv()
            intrinsics_str: str = self.ws.recv()

            try:
                intrinsics_arr = [float(i) for i in intrinsics_str.split(",")]
                self.intrinsics = np.array([
                    [intrinsics_arr[0], 0, intrinsics_arr[2]],
                    [0, intrinsics_arr[1], intrinsics_arr[3]],
                    [0, 0, 1]
                ])
                """
                width=256 height=192 bytesPerRow=1024 pixelFormat=fdep
                """
                img: np.ndarray = np.frombuffer(im, dtype=np.float32)
                self.curr_image = np.rot90(img.reshape((192, 256)), k=-1)
            except Exception as e:
                self.logger.error(f"Failed to decode image: {e}")
        except Exception as e:
            self.logger.error(f"Failed to get image: {e}")

    def run_in_series(self, **kwargs):
        self.receive()
