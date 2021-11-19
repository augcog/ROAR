from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.utilities_module.data_structures_models import Rotation, Location, Transform
import cv2
from collections import deque
from ROAR.utilities_module.udp_multicast_communicator import UDPMulticastCommunicator
from ROAR.control_module.udp_pid_controller import UDP_PID_CONTROLLER

class UDPMultiCastAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.udp_multicast = UDPMulticastCommunicator(mcast_group="224.1.1.1",
                                                      mcast_port=5004,
                                                      threaded=True,
                                                      update_interval=0.1)
        self.add_threaded_module(self.udp_multicast)
        self.controller = UDP_PID_CONTROLLER(agent=self, distance_to_keep=1, center_x=-0.3)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        self_name = "Xingyu"
        self_data = self.vehicle.to_array()
        # location x, y, z; rotation roll, pitch, yaw; velocity x, y, z; acceleration x, y, z
        self.udp_multicast.send_msg(f"{self_name},{','.join(map(str, self_data))}")
        if len(self.udp_multicast.recv_data) > 7 and self.udp_multicast.recv_data[0] != self_name:
            return VehicleControl(throttle=0, steering=0)
            control = self.controller.run_in_series(target_point=self.udp_multicast.recv_data[1:], self_point=self_data)
            return control

        # if 1==1:
        #     fake_data = [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0]
        #     control = self.controller.run_in_series(target_point=fake_data, self_point=self_data)
        #     print(control)
        #     return control
        self.logger.info("No other cars found")
        return VehicleControl(throttle=0, steering=0)
