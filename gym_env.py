# Standard packages
import math, os, errno, random
#Anaconda packages
import gym
import numpy as np
# Other scripts in repo
from util.settings import *
from util.game import TicTacToe 


class TicTacToc_Env(gym.Env):
    """
    Custom Environment using the gym interface for the Tic-Tac-Toe game
    Agent is trained against a previous version of itself or another agent.
    If no agent (previous version or otherwise) is passes to the environment,
    the environment uses an opponent that chooses a random valid action.

    This environment is meant to be used with agents that can mask non-valid 
    actions such as an action there there is already a mark (x or o). The 
    'get_available_actions' method returns a list of actions that are valid 
    (meaning the current postion is empty and the agent can make a mark in that 
    location in the form of positions 0-8 where positions 0-2 are the top row of 
    the game board, positions 3-5 are the middle row, and positions 6-8 are the 
    bottom row.  empty. Use this method to create an action mask within the agent. 
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, game, agent_mark = 'X'):
        # admin settings
        self.game = game
        self.win_reward = WIN_REWARD
        self.tie_reward = TIE_REWARD
        self.playing_human = False
        if agent_mark == 'X':
            self.agent_side = 1
        else:
            self.agent_side = 2

        # Gym settings
        self.size_of_board = len(self.game.board[0]) * len(self.game.board[0])
        values_on_board = list([3]* self.size_of_board)
        self.action_space = gym.spaces.Discrete(self.size_of_board)
        #self.observation_space = gym.spaces.MultiDiscrete([2, *values_on_board])
        self.observation_space = gym.spaces.Dict({
            'action_mask': gym.spaces.Box(0, 1, shape=(self.size_of_board,)),
            'observations': gym.spaces.Box(0, 3, shape=(self.size_of_board + 1,))
        })

        # Training Opponent AI
        if OPPONENT == 'random':
            self.opponent_type = 'random'
            self.opponent = None
        else:
            if os.path.exists(OPPONENT):
                self.opponent_type = 'AI'
                self.opponent = OPPONENT
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), opponent)

    def get_available_actions(self) ->list:
        ''' 
        Returns a list empty positions in the form positions 0-8 where positions
        0-2 are the top row of the game board, positions 3-5 are the middle row, 
        and positions 6-8 are the bottom row.
        '''
        i, j = np.where(self.game.board == 0)
        avail_slots = [i * 3 + j for i, j in list(zip(i,j))]
        return avail_slots

    def mask_fn(self) -> np.array:
        avail_actions = self.get_available_actions()
        #valid_action_mask = [x in avail_actions for x in range(self.size_of_board)]
        valid_action_mask = [1 if x in avail_actions else 0 for x in range(self.size_of_board)]
        return np.asarray(valid_action_mask)

    def _change_actionNum_to_boardPos(self, actionNum) -> tuple:
        '''
        converts an action 0-8 into a tuple in the form of (row, col)
        '''
        i = math.floor(actionNum/3)
        j = actionNum%3
        return (i,j)

    def _get_next_board(self, action) -> np.array:
        '''
        Takes an action in the form of (row, col), makes the players mark (x or o)
        in that location and returns the next board in the form of a np.array.
        '''
        next_board = self.game.board.copy()
        next_board[action] = self.game.player_now
        return next_board

    def _make_state(self) -> np.array:
        '''
        Takes in the current game board in the form of a np.array with i rows 
        and j columns and flattens the array to return the flattened state.
        '''
        state = [self.game.player_now - 1]
        for element in self.game.board.flatten():
            state.append(int(element))
        return np.array(state)

    def _get_reward(self) -> [int, bool, bool]:
        '''
        Checks if the current player has won or tied the game. A win is when
        the player has three marks in a row and a tie is when there is no more 
        valid moves (the board is full). Returns a numeric reward value and a
        boolean value expressing if the game is won or tied.
        '''
        reward = 0
        win = self.game.check_win()
        tie = self.game.check_tie()
        if win:
            reward = self.win_reward
        elif tie:
            reward = self.tie_reward
        return reward, win, tie

    def _get_opponent_action(self) -> None:
        if self.opponent_type == 'random':
            avail_actions = self.get_available_actions()
            action = random.choice(avail_actions)
            action_pos = self._change_actionNum_to_boardPos(action)
        else:
            valid_action_array = self.mask_fn()
            state = self._make_state()
            action = self.opponent.predict(state, action_masks=valid_action_array)
            action_pos = self._change_actionNum_to_boardPos(action)
        # Make action mark on board and return new board
        self.game.board = self._get_next_board(action_pos)
        # change play back to agent
        self.game.change_player()

    def _fix_action_mask(self, obs):
        # Fix action-mask: Everything larger 0.5 is 1.0, everything else 0.0.
        self.valid_actions = np.round(obs["action_mask"])
        obs["action_mask"] = self.valid_actions

    def step(self, action) -> [list, int, bool, dict]:
        # Get next state based on action
        avail_actions = self.get_available_actions()
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
                self._get_opponent_action()
                reward, win, tie = self._get_reward()
                if win or tie:
                    done = True
            state = self._make_state()
        # If action is not allowed (move where a mark is already on board), return current state and bad reward
        else:
            state = self._make_state()
            reward = BAD_ACTION_REWARD
            done = True
        info = {}

        obs = {
           "action_mask": self.mask_fn(),
           "observations": state
        }
        # obs = state

        self.reward = reward
        self.done = done

        self._fix_action_mask(obs)
        return obs, reward, done, info

    def reset(self) -> list:
        self.game.board = np.zeros((3,3))
        if self.game.player_now != self.agent_side:
            self._get_opponent_action()
            pass
        state = self._make_state()
        obs = {
           "action_mask": self.mask_fn(),
           "observations": state
        }
        #obs = state
        self._fix_action_mask(obs)
        return obs

if __name__ == '__main__':
    game = TicTacToe()
    env = TicTacToc_Env(game)

    obs = env.reset()
    count = 0
    done = False
    while not done:
        #action = env.action_space.sample()
        avail_actions = env.get_available_actions()
        action = random.choice(avail_actions)
        obs, reward, done, info = env.step(action)
        count +=1
        if count > 15:
            done = True
        pass