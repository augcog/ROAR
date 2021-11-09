import time

import numpy as np

from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
<<<<<<< HEAD
import torch
import torch.nn as nn


class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.25),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=404736, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, input):
        input = input.view(input.size(0), 1, 600, 800)
        output = self.conv_layers(input)
        output = torch.flatten(output)
        self.linear_layers[0].forward(output)
        output = self.linear_layers(output)
        return output
=======
import cv2
from collections import deque
>>>>>>> 0e653a72edbe678087e156f2b805cc69d6cb3c23


class ForwardOnlyAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
<<<<<<< HEAD
        self.model = torch.load("/home/michael/Desktop/projects/ROAR/misc/model.h5")
        self.model.training = False
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
=======
        self.log = deque(maxlen=100)
        self.start = time.time()
>>>>>>> 0e653a72edbe678087e156f2b805cc69d6cb3c23

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        if self.front_depth_camera.data is not None:
<<<<<<< HEAD
            obs = torch.tensor(self.front_depth_camera.data)
            obs = torch.unsqueeze(obs, 0)
            obs = obs.float().to(self.device)
            steering = self.model(obs)
            return VehicleControl(throttle=0.4, steering=steering.data.item())
=======
            cv2.imshow("depth", self.front_depth_camera.data)
            cv2.waitKey(1)
>>>>>>> 0e653a72edbe678087e156f2b805cc69d6cb3c23
        return VehicleControl(throttle=0.4, steering=0)
