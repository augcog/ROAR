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
        if self.curr_image is not None:
            cv2.imwrite((
                                self.dir_path / f"{self.name}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')}.jpg").as_posix(),
                        self.curr_image)

    def __init__(self, host, port, show=False, resize: Optional[Tuple] = None,
                 name: str = "depth_cam", threaded: bool = True,
                 should_record: bool = False, dir_path: Path = Path("./data/images"),
                 update_interval: float = 0.5):
        super().__init__(threaded=threaded, name=name, update_interval=update_interval)
        self.logger = logging.getLogger(f"{self.name} server on [{host}:{port}]")
        self.host = host
        self.port = port
        self.ws = None

        self.resize = resize
        self.show = show

        self.curr_image: Optional[np.ndarray] = None
        self.should_record = should_record
        self.dir_path = dir_path / f"{self.name}"
        if self.dir_path.exists() is False:
            self.dir_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"{name} initialized")

    def receive(self):
        try:
            self.ws = create_connection(f"ws://{self.host}:{self.port}/{self.name}", timeout=0.1)
            result = self.ws.recv()
            try:
                """
                width=256 height=192 bytesPerRow=1024 pixelFormat=fdep
                """
                img: np.ndarray = np.frombuffer(result, dtype=np.float32)
                self.curr_image = np.rot90(img.reshape((192, 256)), k=-1)
            except Exception as e:
                pass
                # self.logger.error(f"Failed to decode image: {e}")
        except Exception as e:
            # self.logger.error(f"Failed to get image: {e}")
            pass

    def run_in_series(self, **kwargs):
        self.receive()
