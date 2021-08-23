import logging
from websocket import create_connection
from typing import List, Optional, Tuple, List
import cv2
import numpy as np
from pathlib import Path
from ROAR.utilities_module.module import Module

import datetime


class RGBCamStreamer(Module):
    def save(self, **kwargs):
        pass

    def __init__(self, host, port, resize: Optional[Tuple] = None,
                 name: str = "world_cam", threaded: bool = True,
                 update_interval: float = 0.5):
        super().__init__(threaded=threaded, name=name, update_interval=update_interval)

        self.logger = logging.getLogger(f"{self.name} server on [{host}:{port}]")
        self.host = host
        self.port = port
        self.ws = None

        self.resize = resize

        self.curr_image: Optional[np.ndarray] = None
        self.logger.info(f"{name} initialized")

    def receive(self):
        try:
            self.ws = create_connection(f"ws://{self.host}:{self.port}/{self.name}", timeout=0.1)
            img = self.ws.recv()
            # intrinsics = self.ws.recv()
            try:
                img = np.frombuffer(img, dtype=np.uint8)
                self.curr_image = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)[:, :, :3]
                # intrinsics = np.frombuffer(img, dtype=np.float64)
            except Exception as e:
                self.logger.error(f"Failed to decode image: {e}")
        except Exception as e:
            self.logger.error(f"Failed to get image: {e}")
            self.curr_image = None

    def run_in_series(self, **kwargs):
        self.receive()


if __name__ == '__main__':
    ir_image_server = RGBCamStreamer(host="10.142.143.48", port=8005, name="world_cam")
    ir_image_server.run_in_series()
