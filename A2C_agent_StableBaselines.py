# Standard libraries
import os, errno
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
def make_env(opponent) -> gym.Env:
    game = TicTacToe()
    env = TicTacToc_Env(game, opponent)
    #env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    env = ActionMasker(env, env.mask_fn)  # Wrap to enable masking
    #vec_env = make_vec_env(env)
    #env = DummyVecEnv([lambda: TicTacToc_Env(game, playing_human = False)])
    check_env(env)
    return env

#def mask_fn(env: gym.Env) -> np.ndarray:
#    return np.asarray(env.mask_fn())

# Make and train A2C agent
def train_new_agent(env, model, agent_file_name) -> None:
    model.learn(TRAINING_GAMES)
    model.save(agent_file_name)

def progressive_training(env, agent_file_name, evolutions) -> None:
    if os.path.exists(agent_file_name):
        agent = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
        agent.set_parameters(agent_file_name, env = env)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), agent_file_name)
    
    for evolution in range(evolutions):
        opponent = MaskablePPO.set_parameters(agent_file_name, env=env)
        env_AI = make_env(opponent)
        agent.set_env(env_AI)
        agent.learn(TRAINING_GAMES)
        agent.save(agent_file_name)

        print(f'evolution number {evolution} complete')


if __name__ == '__main__':
    env = make_env(opponent = 'random')
    agent = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    train_new_agent(env, agent, agent_file_name)
    progressive_training(env, agent_file_name, evolutions = 10)
    