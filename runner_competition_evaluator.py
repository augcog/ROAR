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


def compute_score(carla_runner: CarlaRunner) -> Tuple[float, int, bool]:
    time_elapsed: float = carla_runner.end_simulation_time - carla_runner.start_simulation_time
    num_collision: int = carla_runner.agent_collision_counter
    lap_completed = True if \
        np.linalg.norm(carla_runner.end_vehicle_position - carla_runner.start_vehicle_position) < 50 else False
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
        carla_runner.start_game_loop(agent=agent, use_manual_control=True)
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
    agent_class = PurePursuitAgent
    num_trials = 2
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
