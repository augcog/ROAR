from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.perception_module.aruco_detector import ArucoDetector
from ROAR.utilities_module.data_structures_models import Rotation, Location, Transform
from typing import Optional
import cv2
import cv2.aruco as aruco
import numpy as np
from collections import deque
from ROAR.control_module.aruco_pid_controller import SimplePIDController

class ArucoFollowingAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.aruco_detector = ArucoDetector(aruco_id=0, agent=self)
        self.controller = SimplePIDController(agent=self, distance_to_keep=1, center_x=-0.3)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        # None if nothing is detected, else, transformation matrix P
        result: Optional[np.ndarray] = self.aruco_detector.run_in_series()
        if self.front_rgb_camera.data is not None:
            cv2.imshow("img", cv2.resize(self.front_rgb_camera.data, (600,800)))
        if result is not None:
            # if i detected the aruco marker
            r: Rotation = self.aruco_detector.rotationMatrixToEulerAngles(result[0:3, 0:3])
            t: Location = Location(x=result[0][3], y=result[1][3], z=result[2][3])
            p: Transform = Transform(location=t, rotation=r)
            control = self.controller.run_in_series(next_waypoint=p)
            return control

        self.logger.info("No Aruco found, NEU")
        # execute previous control
        return VehicleControl(throttle=0, steering=0)
