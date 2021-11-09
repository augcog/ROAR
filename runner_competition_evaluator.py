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
from ROAR.agent_module.michael_pid_agent import PIDAgent


def compute_score(carla_runner: CarlaRunner) -> Tuple[float, int, int]:
    """
    Calculates the score of the vehicle upon completion of the track based on certain metrics
    Args:
        carla_runner ():

    Returns:
        time_elapsed:
        num_collision: number of collisions during simulation
        laps_completed: Number of laps completed

    """
    time_elapsed: float = carla_runner.end_simulation_time - carla_runner.start_simulation_time
    num_collision: int = carla_runner.agent_collision_counter
    laps_completed = 0 if carla_runner.completed_lap_count < 0 else carla_runner.completed_lap_count

    return time_elapsed, num_collision, laps_completed


def run(agent_class, agent_config_file_path: Path, carla_config_file_path: Path,
        num_laps: int = 10) -> Tuple[float, int, int]:
    """
    Run the agent along the track and produce a score based on certain metrics
    Args:
        num_laps: int number of laps that the agent should run
        agent_class: the participant's agent
        agent_config_file_path: agent configuration path
        carla_config_file_path: carla configuration path
    Returns:
        float between 0 - 1 representing scores
    """

    agent_config: AgentConfig = AgentConfig.parse_file(agent_config_file_path)
    carla_config = CarlaConfig.parse_file(carla_config_file_path)

    # hard code agent config such that it reflect competition requirements
    agent_config.num_laps = num_laps
    carla_runner = CarlaRunner(carla_settings=carla_config,
                               agent_settings=agent_config,
                               npc_agent_class=PurePursuitAgent,
                               competition_mode=True,
                               lap_count=num_laps)
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
    agent_class = PIDAgent
    num_trials = 5
    total_score = 0
    num_laps = 2
    table = PrettyTable()
    table.field_names = ["time_elapsed (sec)", "num_collisions", "laps completed"]
    for i in range(num_trials):
        scores = run(agent_class=agent_class,
                     agent_config_file_path=Path("./ROAR/configurations/carla/carla_agent_configuration.json"),
                     carla_config_file_path=Path("./ROAR_Sim/configurations/configuration.json"),
                     num_laps=num_laps)
        table.add_row(scores)
    print(table)


if __name__ == "__main__":
    main()
