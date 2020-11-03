from ROAR.agent_module.legacy_agents.gpd_floodfill_agent import GPDFloodFillAgent
import cv2
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.data_structures_models import MapEntry
from typing import List
import numpy as np


class GPDFloodfillJsonRecordingAgent(GPDFloodFillAgent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig):
        super().__init__(vehicle, agent_settings)
        self.y_pos = 350
        self.center_coord = (self.front_rgb_camera.image_size_x // 2, self.y_pos)
        self.map_history: List[MapEntry] = []

    def run_step(self, sensors_data: SensorsData,
                 vehicle: Vehicle) -> VehicleControl:
        control = super(GPDFloodfillJsonRecordingAgent, self).run_step(sensors_data=sensors_data,
                                                                       vehicle=vehicle)
        if self.gpd_detector.gpd_2d_imposed is not None:
            gpd_2d_imposed = self.gpd_detector.gpd_2d_imposed
            gpd_2d_imposed = cv2.circle(img=gpd_2d_imposed, center=self.center_coord,
                                        radius=2, color=(0, 255, 0), thickness=5)
            whites = np.where(gpd_2d_imposed[350, :] == [255, 255, 255])

            # find the white coords and draw them
            min_x, _ = np.amin(whites, axis=1)
            max_x, _ = np.amax(whites, axis=1)
            left_coord = (min_x, self.y_pos)
            right_coord = (max_x, self.y_pos)
            gpd_2d_imposed = cv2.circle(img=gpd_2d_imposed, center=left_coord,
                                        radius=2, color=(0, 255, 0), thickness=5)
            gpd_2d_imposed = cv2.circle(img=gpd_2d_imposed, center=right_coord,
                                        radius=2, color=(0, 255, 0), thickness=5)
            cv2.imshow("gpd 2d", gpd_2d_imposed)
            cv2.waitKey(1)

            # transpose white cords into world cooridnate
            left_depth = self.front_depth_camera.data[left_coord[1]][left_coord[0]] * 1000
            right_depth = self.front_depth_camera.data[right_coord[1]][right_coord[0]] * 1000
            # print(left_depth, right_depth, left_coord, right_coord)

        return control
