import logging
from ROAR.utilities_module.module import Module
from ROAR.utilities_module.vehicle_models import VehicleControl
from typing import Optional, List
from pathlib import Path
from websocket import create_connection
import requests


class ControlStreamer(Module):
    def save(self, **kwargs):
        self.file.write(self.control.record())

    def __init__(self, host: str, port: int, name: str = "transform",
                 threaded: bool = True,
                 should_record: bool = True, file_path: Path = Path("./data/transforms.txt")):
        super().__init__(threaded=threaded, name=name)
        self.logger = logging.getLogger(f"{self.name} server [{host}:{port}]")
        self.host = host
        self.port = port
        self.control: VehicleControl = VehicleControl()
        self.ws_tx = None
        self.ws_rx = None
        self.should_record = should_record
        self.file_path: Path = file_path
        self.file = open(self.file_path.as_posix(), "a")
        self.control_history: List[str] = []
        self.logger.info(f"{name} initialized")

    def send(self, vehicle_control: VehicleControl):
        try:
            param = {
                "throttle": vehicle_control.throttle,
                "steering": vehicle_control.steering
            }

            respond = requests.post(f"http://{self.host}:{self.port}/{self.name}_rx", json=param, timeout=1)
        except requests.exceptions.Timeout:
            self.logger.error("Send Timed out")
        except requests.exceptions.ConnectionError:
            self.logger.error("Unable to connect")
        # self.logger.info(f"{param}, {respond.status_code}")

    def receive(self):
        try:
            self.ws_rx = create_connection(f"ws://{self.host}:{self.port}/{self.name}_tx")
            result: bytes = self.ws_rx.recv()
            try:
                self.control = VehicleControl.fromBytes(result)
                if self.should_record:
                    self.control_history.append(self.control)
            except Exception as e:
                pass
                # self.logger.error(f"Failed to parse data {e}. {result}")

        except Exception as e:
            pass
            # self.logger.error(f"Failed to get data: {e}")

    def run_in_series(self, **kwargs):
        self.receive()

    def shutdown(self):
        super(ControlStreamer, self).shutdown()
        self.file.close()
