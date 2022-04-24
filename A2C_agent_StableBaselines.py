# Standard libraries
import os
# Anaconda libraries
import gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
# from stable_baselines3 import A2C
# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env


# Other scripts in this repo
from util.settings import *
from util.game import TicTacToe
from gym_env import TicTacToc_Env
#from ParametricActionsModel import ParametricActionsModel

# Agent name for saving
agent_folder = 'agents'
agent_file_name = os.path.join(agent_folder, 'AC2_Agent_TicTacToe')

# Make environment
def make_env(playing_human = False):
    game = TicTacToe()
    env = TicTacToc_Env(game, playing_human)
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    #vec_env = make_vec_env(env)
    #env = DummyVecEnv([lambda: TicTacToc_Env(game, playing_human = False)])
    check_env(env)
    return env


def mask_fn(env) -> np.ndarray:
    avail_actions = env.get_available_actions(env.game.board)
    valid_action_mask = [x in avail_actions for x in range(env.size_of_board)]
    return valid_action_mask


# Make and train A2C agent
def train_new_agent(env):
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model.learn(TRAINING_GAMES)
    return model


if __name__ == '__main__':
    env = make_env()
    agent = train_new_agent(env)
    agent.save(agent_file_name)