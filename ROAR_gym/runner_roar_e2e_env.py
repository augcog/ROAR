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
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.agent import Agent
from ROAR.agent_module.rl_depth_e2e_agent import RLDepthE2EAgent

from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.policies import CnnPolicy

from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, CallbackList
from util import find_latest_model, CustomMaxPoolCNN

try:
    from ROAR_Gym.envs.roar_env import LoggingCallback
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import LoggingCallback
from datetime import datetime


def main():
    # Set gym-carla environment
    agent_config = AgentConfig.parse_file(Path("configurations/agent_configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("configurations/carla_configuration.json"))

    params = {
        "agent_config": agent_config,
        "carla_config": carla_config,
        "ego_agent_class": RLDepthE2EAgent
    }

    model_dir_path = Path("./output/e2e_2")

    policy_kwargs = dict(
        features_extractor_class=CustomMaxPoolCNN,
        features_extractor_kwargs=dict(features_dim=256)
    )

    training_kwargs = dict(
        learning_rate=0.001,
        buffer_size=5000,
        batch_size=64,
        learning_starts=10000,
        gamma=0.95,
        train_freq=(4, "step"),
        gradient_steps=1,
        target_update_interval=1000,
        exploration_initial_eps=1,
        exploration_final_eps=0.1,
        exploration_fraction=0.2,
        seed=1,
        device="cuda",
        verbose=1,
        tensorboard_log=(Path(model_dir_path) / "tensorboard").as_posix()
    )

    env = gym.make('roar-e2e-v0', params=params)
    env.reset()

    latest_model_path = find_latest_model(model_dir_path)

    if latest_model_path is None:
        model = DQN(CnnPolicy, env=env, policy_kwargs=policy_kwargs, **training_kwargs)
    else:
        model = DQN.load(latest_model_path, env=env, policy_kwargs=policy_kwargs, **training_kwargs)
    print("Model Loaded Successfully")
    logging_callback = LoggingCallback(model=model)
    checkpoint_callback = CheckpointCallback(save_freq=1000, verbose=2, save_path=(model_dir_path/"logs").as_posix())
    event_callback = EveryNTimesteps(n_steps=100, callback=checkpoint_callback)
    callbacks = CallbackList([checkpoint_callback, event_callback, logging_callback])
    model = model.learn(total_timesteps=int(1e10), callback=callbacks, reset_num_timesteps=False)
    # model.save(model_dir_path / f"roar_e2e_model_{datetime.now()}")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt="%H:%M:%S", level=logging.INFO)
    logging.getLogger("Controller").setLevel(logging.ERROR)
    logging.getLogger("SimplePathFollowingLocalPlanner").setLevel(logging.ERROR)
    main()
