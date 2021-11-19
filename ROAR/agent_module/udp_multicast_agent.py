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
        self.name = "car_2"
        self.udp_multicast = UDPMulticastCommunicator(agent=self,
                                                      mcast_group="224.1.1.1",
                                                      mcast_port=5004,
                                                      threaded=True,
                                                      update_interval=0.025,
                                                      name=self.name)
        self.add_threaded_module(self.udp_multicast)
        self.controller = UDP_PID_CONTROLLER(agent=self, distance_to_keep=1, center_x=-0.3)
        self.car_to_follow = "car_1"

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        if self.time_counter % 10 == 0:
            self.udp_multicast.send_current_state()
        # location x, y, z; rotation roll, pitch, yaw; velocity x, y, z; acceleration x, y, z
        if self.udp_multicast.msg_log.get(self.car_to_follow, None) is not None:
            control = self.controller.run_in_series(target_point=self.udp_multicast.msg_log[self.car_to_follow])
            return control
            # return VehicleControl()
        else:
            # self.logger.info("No other cars found")
            return VehicleControl(throttle=0, steering=0)
