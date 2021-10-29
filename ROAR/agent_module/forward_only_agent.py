from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import cv2


class ForwardOnlyAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        if self.front_depth_camera.data is not None:
            cv2.imshow("depth", self.front_depth_camera.data)
            cv2.waitKey(1)
        # self.logger.info(self.vehicle.get_speed(self.vehicle))
        # if self.front_rgb_camera.data is not None:
        #     frame = self.front_rgb_camera.data.copy()
        #     s = frame.shape
        #     height = 3 * s[1] // 4
        #     min_y = s[0] - height
        #     max_y = s[0]
        #     display_view = frame[min_y:max_y, :]
        #     cv2.imshow("front rgb", cv2.resize(display_view, dsize=(1080, 810)))
        #     cv2.waitKey(1)
        return VehicleControl(throttle=0.4, steering=0)
