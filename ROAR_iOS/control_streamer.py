import logging
from ROAR.utilities_module.module import Module
from ROAR.utilities_module.vehicle_models import VehicleControl
from typing import Optional, List
from pathlib import Path
from websocket import create_connection


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
        self.ws = create_connection(f"ws://{self.host}:{self.port}/{name}")
        self.ack = "ack".encode("utf-8")
        self.should_record = should_record
        self.file_path: Path = file_path
        self.file = open(self.file_path.as_posix(), "a")
        self.control_history: List[str] = []
        self.logger.info(f"{name} initialized")

    def send(self, vehicle_control: VehicleControl):
        self.ws = create_connection(f"ws://{self.host}:{self.port}/{self.name}")
        self.ws.send(vehicle_control.record().encode("utf-8"))

    def receive(self):
        try:
            self.ws = create_connection(f"ws://{self.host}:{self.port}/{self.name}")
            result: bytes = self.ws.recv()
            try:
                self.control = VehicleControl.fromBytes(result)
                if self.should_record:
                    self.control_history.append(self.control)
            except Exception as e:
                self.logger.error(f"Failed to parse data {e}. {result}")

        except Exception as e:
            self.logger.error(f"Failed to get data: {e}")

    def run_in_series(self, **kwargs):
        while True:
            self.receive()

    def run_in_threaded(self, **kwargs):
        self.run_in_series()

    def shutdown(self):
        super(ControlStreamer, self).shutdown()
        self.file.close()
