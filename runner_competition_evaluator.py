import logging, warnings
import numpy as np
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR.configurations.configuration import Configuration as AgentConfig
from pathlib import Path
from ROAR.agent_module.pure_pursuit_agent \
    import PurePursuitAgent
from ROAR_Sim.carla_client.carla_runner import CarlaRunner
from typing import Tuple
from prettytable import PrettyTable
from ROAR.agent_module.jAM1AgentOld import JAM1Agent_old
from ROAR.agent_module.jAM3AgentOld import JAM3Agent_old



def compute_score(carla_runner: CarlaRunner, min_bounding_box = np.array([0,-2,30]), max_bounding_box = np.array([60,2,60])) -> Tuple[float, int, bool]:
    """
    Calculates the score of the vehicle upon completion of the track based on certain metrics
    Args:
        carla_runner ():
        min_bounding_box ():
        max_bounding_box ():

    Returns:
        time_elapsed:
        num_collision: number of collisions during simulation
        lap_completed: True if vehicle reaches the finish bounding box

    """
    time_elapsed: float = carla_runner.end_simulation_time - carla_runner.start_simulation_time
    num_collision: int = carla_runner.agent_collision_counter

    lower_diff = carla_runner.end_vehicle_position - min_bounding_box
    upper_diff = max_bounding_box - carla_runner.end_vehicle_position
    print('lower diff = ',lower_diff)
    print('upper diff = ',upper_diff)

    lower_check = [True if n > 0 else False for n in lower_diff]
    upper_check = [True if n > 0 else False for n in upper_diff]
    lap_completed = all(lower_check) and all(upper_check)

    return time_elapsed, num_collision, lap_completed



def run(agent_class, agent_config_file_path: Path, carla_config_file_path: Path) -> Tuple[float, int, bool]:
    """
    Run the agent along the track and produce a score based on certain metrics
    Args:
        agent_class: the participant's agent
        agent_config_file_path: agent configuration path
        carla_config_file_path: carla configuration path
    Returns:
        float between 0 - 1 representing scores
    """
    agent_config = AgentConfig.parse_file(agent_config_file_path)
    carla_config = CarlaConfig.parse_file(carla_config_file_path)

    carla_runner = CarlaRunner(carla_settings=carla_config,
                               agent_settings=agent_config,
                               npc_agent_class=PurePursuitAgent,
                               competition_mode=True,
                               max_collision=3)
    try:
        my_vehicle = carla_runner.set_carla_world()
        agent = agent_class(vehicle=my_vehicle, agent_settings=agent_config)
        carla_runner.start_game_loop(agent=agent, use_manual_control=False)
        return compute_score(carla_runner)
    except Exception as e:
        print(f"something bad happened during initialization: {e}")
        carla_runner.on_finish()
        logging.error(f"{e}. Might be a good idea to restart Server")
        return 0, 0, False


def suppress_warnings():
    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(name)s '
                               '- %(message)s',
                        level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    warnings.simplefilter("ignore")
    np.set_printoptions(suppress=True)


def main():
    suppress_warnings()
    agent_class = JAM1Agent_old
    num_trials = 3
    total_score = 0
    table = PrettyTable()
    table.field_names = ["time_elapsed (sec)", "num_collisions", "lap_completed (T/F)"]
    for i in range(num_trials):
        scores = run(agent_class=agent_class,
                     agent_config_file_path=Path("./ROAR_Sim/configurations/agent_configuration.json"),
                     carla_config_file_path=Path("./ROAR_Sim/configurations/configuration.json")
                     )
        table.add_row(scores)
    print(table)


if __name__ == "__main__":
    main()
