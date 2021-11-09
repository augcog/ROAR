import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from typing import Tuple, List

import carla
import numpy as np

from Bridges.carla_bridge import CarlaBridge
from ROAR.agent_module.agent import Agent
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle
from ROAR_Desktop.utilities.runnable import Runnable
from ROAR_Sim.carla_client.util.hud import HUD
from ROAR_Sim.carla_client.util.world import World
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR_Desktop.ROAR_GUI.control.gui_utilities import KeyboardControl


class CarlaRunner(Runnable):
    def __init__(self, carla_setting: CarlaConfig,
                 agent_setting: AgentConfig,
                 bridge: CarlaBridge,
                 keyboard_control: KeyboardControl,
                 npc_agent_class, competition_mode=False, max_collision=1000):
        super().__init__(environment_setting=carla_setting,
                         agent_setting=agent_setting,
                         keyboard_control=keyboard_control,
                         bridge=bridge)

        self.NPCAgentClass = npc_agent_class
        self.competition_mode = competition_mode
        self.max_collision = max_collision

        self.world = None
        self.client = None
        self.display = None
        self.agent = None

        self.npc_agents: Dict[npc_agent_class, Any] = {}
        self.agent_collision_counter = 0

        self.competition_mode = competition_mode
        self.max_collision = max_collision
        self.start_simulation_time: Optional[float] = None
        self.start_vehicle_position: Optional[np.array] = None
        self.end_simulation_time: Optional[float] = None
        self.end_vehicle_position: Optional[np.array] = None

        self.logger = logging.getLogger(__name__)
        self.timestep_counter = 0

    def setup(self):
        try:
            self.client = carla.Client(self.environment_setting.host,
                                       self.environment_setting.port)
            self.client.set_timeout(self.environment_setting.timeout)
            self.logger.debug(f"Setting HUD")
            hud = HUD(self.environment_setting.width, self.environment_setting.height)

            self.logger.debug("Setting up world")
            self.world = World(carla_world=self.client.get_world(), hud=hud,
                               carla_settings=self.environment_setting,
                               agent_settings=self.agent_setting)

            if self.environment_setting.should_spawn_npcs:
                self.spawn_npcs()

            self.agent = self.bridge.convert_vehicle_from_source_to_agent(self.world.player)
        except Exception as e:
            print(e)

    def start_game_loop(self, use_manual_control=False):
        pass

    def on_finish(self):
        pass

    def convert_data(self) -> Tuple[SensorsData, Vehicle]:
        pass

    def spawn_npcs(self):
        # parse npc file
        npc_config_file_path = Path(self.environment_setting.npc_config_file_path)
        assert npc_config_file_path.exists(), f"NPC file path {npc_config_file_path.as_posix()} does not exist"
        npc_configs = json.load(npc_config_file_path.open('r'))

        npc_configs: List[AgentConfig] = [AgentConfig.parse_obj(config) for config in npc_configs]

        self.world.spawn_npcs(npc_configs)
        self.npc_agents = {
            self.NPCAgentClass(vehicle=actor, agent_settings=npc_config): actor for actor, npc_config in
            self.world.npcs_mapping.values()
        }
