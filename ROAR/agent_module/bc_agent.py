from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import torch
import torch.nn as nn


class CarModel(nn.Module):
    def __init__(self):
        super(CarModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(24, 48, 5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(48, 96, 5),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.LazyLinear(out_features=100),
            nn.Dropout(),
            nn.ELU(),

            nn.Linear(in_features=100, out_features=50),
            nn.Dropout(),
            nn.ELU(),

            nn.Linear(in_features=50, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, input):
        input = torch.reshape(input, (1, 1, 600, 800))
        output = self.conv_layers(input)
        output = output.flatten()
        output = self.linear_layers(output)
        return output


class BCAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.model = torch.load("/home/michael/Desktop/projects/ROAR/misc/data/best_model_oct_10.h5")
        self.model.training = False
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        if self.front_depth_camera.data is not None:
            obs = torch.tensor(self.front_depth_camera.data)
            obs = torch.unsqueeze(obs, 0)
            obs = obs.float().to(self.device)
            steering = self.model(obs).data.item()
            return VehicleControl(throttle=0.5, steering=steering)
        return VehicleControl(throttle=0.5, steering=0)
