import warnings
import logging
from typing import Optional, Dict

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("numpy").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
import os
import sys
from pathlib import Path

sys.path.append(Path(os.getcwd()).parent.as_posix())
import gym
import ROAR_Gym
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.agent import Agent
from ROAR.agent_module.rl_pid_agent import RLPIDAgent
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import DDPG
from datetime import datetime
from stable_baselines.common.callbacks import CheckpointCallback, EveryNTimesteps, CallbackList

try:
    from ROAR_Gym.envs.roar_env import LoggingCallback
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import LoggingCallback


def main():
    # Set gym-carla environment
    agent_config = AgentConfig.parse_file(Path("configurations/agent_configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("configurations/carla_configuration.json"))

    params = {
        "agent_config": agent_config,
        "carla_config": carla_config,
        "ego_agent_class": RLPIDAgent
    }

    env = gym.make('roar-pid-v0', params=params)
    env.reset()

    model_params: dict = {
        "verbose": 1,
        "render": True,
        "tensorboard_log": "./output/tensorboard/pid/"
    }
    latest_model_path = find_latest_model(Path(os.getcwd()))
    if latest_model_path is None:
        model = DDPG(LnMlpPolicy, env=env, **model_params)  # full tensorboard log can take up space quickly
    else:
        model = DDPG.load(latest_model_path, env=env, **model_params)
        model.render = True
        model.tensorboard_log = "./output/tensorboard/pid/"

    logging_callback = LoggingCallback(model=model)
    checkpoint_callback = CheckpointCallback(save_freq=1000, verbose=2, save_path='./output/logs')
    event_callback = EveryNTimesteps(n_steps=100, callback=checkpoint_callback)
    callbacks = CallbackList([checkpoint_callback, event_callback, logging_callback])
    model = model.learn(total_timesteps=int(1e10), callback=callbacks, reset_num_timesteps=False)
    model.save(f"pid_ddpg_{datetime.now()}")


def find_latest_model(root_path: Path) -> Optional[Path]:
    import os
    from pathlib import Path
    logs_path = (root_path / "output" / "logs")
    if logs_path.exists() is False:
        print(f"No previous record found in {logs_path}")
        return None
    paths = sorted((root_path / "output" / "logs").iterdir(), key=os.path.getmtime)
    paths_dict: Dict[int, Path] = {
        int(path.name.split("_")[2]): path for path in paths
    }
    latest_model_file_path: Optional[Path] = paths_dict.get(max(paths_dict.keys()), None)
    return latest_model_file_path


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt="%H:%M:%S", level=logging.INFO)
    logging.getLogger("Controller").setLevel(logging.ERROR)
    logging.getLogger("SimplePathFollowingLocalPlanner").setLevel(logging.ERROR)
    main()

