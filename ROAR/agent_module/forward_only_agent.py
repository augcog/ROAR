from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import torch


# from ROAR.utilities_module.utilities import NvidiaModel

class ForwardOnlyAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.model = torch.load("/home/michael/Desktop/projects/ROAR/misc/model.h5")
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        if self.front_depth_camera.data is not None:
            obs = torch.tensor(self.front_depth_camera.data)
            obs = torch.unsqueeze(obs, 0)
            obs = obs.float().to(self.device)
            steering = self.model(obs)
            return VehicleControl(throttle=0.4, steering=steering.data.item())
        control = VehicleControl(throttle=0.4, steering=0)
        return control
