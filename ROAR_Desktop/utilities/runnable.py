from abc import ABC, abstractmethod
from ROAR.configurations.configuration import Configuration as AgentConfig
import logging
from Bridges.bridge import Bridge
from ROAR.agent_module.agent import Agent
from typing import Tuple
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle
from ROAR_Desktop.ROAR_GUI.control.gui_utilities import KeyboardControl

class Runnable(ABC):
    def __init__(self,
                 environment_setting,
                 agent_setting: AgentConfig,
                 keyboard_control: KeyboardControl,
                 bridge: Bridge):
        self.environment_setting = environment_setting
        self.agent_setting = agent_setting
        self.bridge = bridge
        self.controller = keyboard_control
        self.logger = logging.getLogger("Runnable")

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def start_game_loop(self, use_manual_control=False):
        pass

    @abstractmethod
    def on_finish(self):
        pass

    @abstractmethod
    def convert_data(self) -> Tuple[SensorsData, Vehicle]:
        pass
