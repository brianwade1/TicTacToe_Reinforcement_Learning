# Standard packages
import math
#Anaconda packages
import gym
import numpy as np
# Other scripts in repo
from util.settings import *
from util.game import TicTacToe 

class TicTacToc_Env(gym.Env):
    """Custom Environment using the gym interface for the Tic-Tac-Toe game"""
    metadata = {'render_modes': ['human']}

    def __init__(self, game, playing_human = False):

        # admin settings
        self.game = game
        self.win_reward = WIN_REWARD
        self.tie_reward = TIE_REWARD
        self.playing_human = playing_human

        # Gym settings
        self.size_of_board = len(self.game.board[0]) * len(self.game.board[0])
        values_on_board = list([3]* self.size_of_board)
        self.action_space = gym.spaces.Discrete(self.size_of_board)
        self.observation_space = gym.spaces.MultiDiscrete([2, *values_on_board])

    def get_available_actions(self, board) ->list:
        i, j = np.where(board == 0)
        avail_slots = [i * 3 + j for i, j in list(zip(i,j))]
        return avail_slots

    def _change_actionNum_to_boardPos(self, actionNum):
        i = math.floor(actionNum/3)
        j = actionNum%3
        return (i,j)

    def _get_next_board(self, action):
        next_board = self.game.board.copy()
        next_board[action] = self.game.player_now
        return next_board

    def _make_state(self):
        state = [self.game.player_now - 1]
        for element in self.game.board.flatten():
            state.append(int(element))
        return np.array(state)

    def _get_reward(self):
        reward = 0
        win = self.game.check_win()
        tie = self.game.check_tie()
        if win:
            reward = self.win_reward
        elif tie:
            reward = self.tie_reward
        return reward, win, tie

    def step(self, action):
        # Get next state based on action
        avail_actions = self.get_available_actions(self.game.board)
        # If action is allowable, do action, and check for win/tie
        if action in avail_actions:
            action_pos = self._change_actionNum_to_boardPos(action)
            self.game.board = self._get_next_board(action_pos)
            reward, win, tie = self._get_reward()
            done = False
            if win or tie:
                done = True
            else:
                self.game.change_player()

            state = self._make_state()
        # If action is not allowed (move where a mark is already on board), return current state and bad reward
        else:
            state = self._make_state()
            reward = BAD_ACTION_REWARD
            done = False
        info = {}

        #obs = {
        #    "action_mask": self.action_mask,
        #    "avail_actions": self.action_assignments,
        #    "true_state": state
        #}
        obs = state

        self.reward = reward
        self.done = done

        return obs, reward, done, info

    def reset(self):
        self.game.board = np.zeros((3,3))
        state = self._make_state()
        #obs = {
        #    "action_mask": self.action_mask,
        #    "avail_actions": self.action_assignments,
        #    "true_state": state
        #}
        obs = state
        return obs

if __name__ == '__main__':
    game = TicTacToe()
    env = TicTacToc_Env(game, playing_human = False)

    obs = env.reset()
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        pass