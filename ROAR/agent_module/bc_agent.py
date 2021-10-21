from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from misc.utils import depthToLogDepth
from ROAR.configurations.configuration import Configuration as AgentConfig
import torch
from misc.model import CarModel
import cv2
from collections import deque
import numpy as np

class BCAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.model = torch.load("/home/michael/Desktop/projects/ROAR/misc/data/best_model_oct_10.h5")
        self.model.training = False
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 1
        self.image_queue = deque(maxlen=self.batch_size)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        if self.front_depth_camera.data is not None:
            # self.image_queue.append(cv2.resize(depthToLogDepth(self.front_depth_camera.data),
            #                                    dsize=(80, 60)))
            self.image_queue.append(self.front_depth_camera.data)

        if len(self.image_queue) == self.batch_size:
            obs = torch.tensor(np.array(self.image_queue)).float().to(self.device)
            steerings = self.model(obs).cpu().detach().numpy()
            steering = steerings[-1]
            print(steering)
            return VehicleControl(throttle=0.5, steering=steering)

        return VehicleControl(throttle=0.5, steering=0)
