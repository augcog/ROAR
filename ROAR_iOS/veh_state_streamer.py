import time

import numpy as np
import sys, os
from pathlib import Path
import time

sys.path.append(Path(os.getcwd()).parent.as_posix())
from ROAR_iOS.udp_receiver import UDPStreamer
from ROAR.utilities_module.data_structures_models import Transform, Vector3D


class VehicleStateStreamer(UDPStreamer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transform = Transform()
        self.velocity = Vector3D()
        self.acceleration = Vector3D()

    def run_in_series(self, **kwargs):
        try:
            data = self.recv()
            if data is None:
                return 
            d = [float(s) for s in data.decode('utf-8').split(",")]
            # d = np.frombuffer(data, dtype=np.float32)
            self.transform.location.x = d[0]
            self.transform.location.y = d[1]
            self.transform.location.z = d[2]
            self.transform.rotation.roll = d[3]
            self.transform.rotation.pitch = d[4]
            self.transform.rotation.yaw = d[5]
            self.velocity.x = d[6]
            self.velocity.y = d[7]
            self.velocity.z = d[8]
            self.acceleration.x = d[9]
            self.acceleration.y = d[10]
            self.acceleration.z = d[11]

        except Exception as e:
            self.logger.error(e)
if __name__ == '__main__':
    streamer = VehicleStateStreamer(ios_address="10.0.0.26",
                                             port=8003,
                                             name="VehicleStateStreamer",
                                             update_interval=0.025,
                                             threaded=True)
    while True:
        streamer.run_in_series()
        print(streamer.transform, streamer.velocity)