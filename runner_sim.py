import logging
from pathlib import Path
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR_Sim.carla_client.carla_runner import CarlaRunner
from ROAR.agent_module.pure_pursuit_agent import PurePursuitAgent
from ROAR.configurations.configuration import Configuration as AgentConfig


from ROAR.agent_module.jAM1AgentOld import JAM1Agent_old
#from ROAR.agent_module.jAM1Agent import JAM1Agent
from ROAR.agent_module.jAM2Agent import JAM2Agent
from ROAR.agent_module.jAM3AgentOld import JAM3Agent_old
from ROAR.agent_module.jAM3Agent import JAM3Agent
from ROAR.agent_module.occupancy_map_agent import OccupancyMapAgent

from ROAR.agent_module.special_agents.json_waypoint_generating_agent import JSONWaypointGeneratingAgent
from ROAR.agent_module.special_agents.waypoint_generating_agent import WaypointGeneratigAgent
from ROAR.agent_module.pid_agent import PIDAgent
from ROAR.agent_module.lqr_agent import LQRAgent

from ROAR.agent_module.special_agents.recording_agent import RecordingAgent
from ROAR.agent_module.occu_map_demo_driving_agent import OccuMapDemoDrivingAgent



def main():
    """Starts game loop"""
    agent_config = AgentConfig.parse_file(Path("./ROAR_Sim/configurations/agent_configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("./ROAR_Sim/configurations/configuration.json"))

    carla_runner = CarlaRunner(carla_settings=carla_config,
                               agent_settings=agent_config,
                               npc_agent_class=PurePursuitAgent)
    try:
        my_vehicle = carla_runner.set_carla_world()


        #agent = PIDAgent(vehicle=my_vehicle, agent_settings=agent_config)
        #agent = OccupancyMapAgent(vehicle=my_vehicle, agent_settings=agent_config)
        #agent = PurePursuitAgent(vehicle=my_vehicle, agent_settings=agent_config)

        #agent = JAM1Agent_old(vehicle=my_vehicle, agent_settings=agent_config)  # *** roll controller
        #agent = JAM1Agent(vehicle=my_vehicle, agent_settings=agent_config)
        #agent = JAM2Agent(vehicle=my_vehicle, agent_settings=agent_config)
        agent = JAM3Agent_old(vehicle=my_vehicle, agent_settings=agent_config) # *** bstanley
        #agent = JAM3Agent(vehicle=my_vehicle, agent_settings=agent_config)

        # *** use to record new waypoints ***
        # waypointrecord = agent.bstanley_controller.blat_stanley_controller.waypointrecord
        # np.save("James_waypoints", np.array(waypointrecord))

        agent = RecordingAgent(vehicle=my_vehicle, agent_settings=agent_config)

        #carla_runner.start_game_loop(agent=agent, use_manual_control=True)#*******True for manual control, False auto
        carla_runner.start_game_loop(agent=agent, use_manual_control=False)  # *******True for manual control, False auto




    except Exception as e:
        logging.error(f"Something bad happened during initialization: {e}")
        carla_runner.on_finish()
        logging.error(f"{e}. Might be a good idea to restart Server")


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(name)s '
                               '- %(message)s',
                        datefmt="%H:%M:%S",
                        level=logging.DEBUG)
    import warnings
    warnings.filterwarnings("ignore", module="carla")
    main()
