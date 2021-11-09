import sys
from pathlib import Path
import os

print(Path(os.getcwd()).parent.parent.as_posix())
sys.path.append(Path(os.getcwd()).parent.parent.as_posix())
from ROAR_Desktop.ROAR_Server.base_server import ROARServer
import numpy as np
import cv2
from typing import Optional
latest_rgb_image: Optional[np.array] = None
latest_depth_image: Optional[np.array] = None

class RealsenseServer(ROARServer):

    def run(self):
        global latest_rgb_image, latest_depth_image

