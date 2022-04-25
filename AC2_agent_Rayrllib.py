# Standard libraries
import os, errno
# Anaconda libraries
import gym
import numpy as np
# pip imports
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print


# Other scripts in this repo
from util.settings import *
from util.game import TicTacToe
from gym_env import TicTacToc_Env
from ParametricActionsModel import ParametricActionsModel

# Agent name for saving
agent_folder = 'agents'
agent_file_name = os.path.join(agent_folder, 'AC2_Agent_TicTacToe')

# Make environment
def env_creator(env_config):
    #opponent = env_config['opponent']
    game = TicTacToe()
    env = TicTacToc_Env(game)
    return env


if __name__ == '__main__':
    ray.init()

    register_env("TicTacToe", env_creator)
    ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)

    config = {
            "env": "TicTacToe",
            "env_config": {
                "opponent": 'random'
            },
            "model": {
                "custom_model": "pa_model"
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": 0,
            "framework": 'tf'
        }

    stop = {
        "training_iteration": TRAINING_GAMES
    }

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(config)
    game = TicTacToe()
    env = TicTacToc_Env(game)
    trainer = ppo.PPOTrainer(config=ppo_config, env='TicTacToe')
    # run manual training loop and print results after each iteration
    for _ in range(5):
        result = trainer.train()
        print(pretty_print(result))
        # stop training if the target train steps or reward are reached
        if (
            result["training_iteration"] >= TRAINING_GAMES
        ):
            break


    #results = tune.run("PPO", stop=stop, config=config, verbose=1)