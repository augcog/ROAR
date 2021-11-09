from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from misc.utils import depthToLogDepth
from ROAR.configurations.configuration import Configuration as AgentConfig
from collections import deque
import numpy as np
import torch
from misc.model import CarModel
import cv2


class BCAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.batch_size = 1
        self.model = CarModel(batch_size=self.batch_size)
        self.model.load_state_dict(torch.load("/home/michael/Desktop/projects/ROAR/misc/data/best_model_state_dict.h5"))
        self.logger.info("model loaded")
        self.model.training = False
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_queue = deque(maxlen=self.batch_size)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        if self.front_depth_camera.data is not None:
            self.image_queue.append(depthToLogDepth(self.front_depth_camera.data))

        if len(self.image_queue) == self.batch_size:
            obs = torch.tensor(np.array(self.image_queue)).float()
            steerings = self.model(obs).cpu().detach().numpy()
            steering = steerings[-1]
            print(steering)
            return VehicleControl(throttle=0.5, steering=steering)

        return VehicleControl(throttle=0.5, steering=0)
