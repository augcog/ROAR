from ROAR.agent_module.agent import Agent
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.control_module.pid_controller import VehiclePIDController
from ROAR.perception_module.legacy.flood_fill_lane_detector import FloodfillLaneDetector
from ROAR.control_module.pid_controller import PIDParam
import cv2
import numpy as np


class FloodfillBasedLaneFollower(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.controller = VehiclePIDController(agent=self, args_lateral=PIDParam.default_lateral_param(),
                                               args_longitudinal=PIDParam.default_longitudinal_param())
        self.floodfill_lane_detector = FloodfillLaneDetector(agent=self)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        try:
            img = self.floodfill_lane_detector.run_in_series()

            # left, front, right_steering dot img location
            left_dot_coord = (self.front_rgb_camera.image_size_x // 4, 350)
            center_dot_coord = (self.front_rgb_camera.image_size_x // 2, 350)
            right_dot_coord = (self.front_rgb_camera.image_size_x - (self.front_rgb_camera.image_size_x // 4), 350)
            blue = [255, 0, 0]

            left_ok = self._is_equal(img[left_dot_coord[::-1]], blue)
            center_ok = self._is_equal(img[center_dot_coord[::-1]], blue)
            right_ok = self._is_equal(img[right_dot_coord[::-1]], blue)

            result = cv2.circle(img=img, center=left_dot_coord, radius=10,
                                color=(0, 0, 255), thickness=-1)
            result = cv2.circle(img=result, center=center_dot_coord, radius=10,
                                color=(0, 0, 255), thickness=-1)
            result = cv2.circle(img=result, center=right_dot_coord, radius=10,
                                color=(0, 0, 255), thickness=-1)
            cv2.imshow("rgb image", result)
            cv2.waitKey(1)
            straight_throttle, turning_throttle, left_steering, right_steering = 0.18, 0.15, -0.4, 0.4
            throttle, steering = 0, 0
            if bool(left_ok) is False:
                # print("GO RIGHT!")
                throttle = turning_throttle
                steering = left_steering
            elif bool(right_ok) is False:
                # print("GO LEFT!")
                throttle = turning_throttle
                steering = right_steering
            elif center_ok:
                throttle, steering = straight_throttle, 0
            # if center_ok:
            #     throttle, steering = 0.5, 0
            # elif left_ok:
            #     throttle = 0.3
            #     steering = -0.5
            # elif right_ok:
            #     throttle = 0.3
            #     steering = 0.5

            # self.logger.info(f"Throttle = {throttle}, steering = {steering}")
            return VehicleControl(throttle=throttle, steering=steering)
        except:
            return VehicleControl()

    @staticmethod
    def _is_equal(arr1, arr2):
        # print(sum(arr1 == arr2))
        return np.alltrue(arr1 == arr2)
