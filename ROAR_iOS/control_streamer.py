from typing import List, Optional, Tuple, List
import cv2
import numpy as np
import sys, os
from pathlib import Path
import time

sys.path.append(Path(os.getcwd()).parent.as_posix())
from ROAR_iOS.udp_receiver import UDPStreamer
from ROAR.utilities_module.vehicle_models import VehicleControl
import struct
MAX_DGRAM = 9600


class ControlStreamer(UDPStreamer):
    def __init__(self, **kwargs):
        super(ControlStreamer, self).__init__(**kwargs)
        self.control_tx = VehicleControl()

    def send(self, control:VehicleControl):
        self.control_tx = control
        string_format = f"{control.throttle},{control.steering}"
        self._send_data(string_format)

    def run_in_series(self, **kwargs):
        pass


if __name__ == '__main__':
    cs = ControlStreamer(pc_port=8004,
                         threaded=False,
                         name="control_streamer")
    cs.connect()
    while True:
        cs.send(VehicleControl(throttle=1, steering=1))
