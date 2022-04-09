# Standard Libraries
import random, os, pickle
#Anaconda Libraries
import numpy as np
# Other scripts
from game import TicTacToe
from settings import *

agent_folder = 'agents'

class Q_Learning_Agent():
    def __init__(self, game, side, playing_human = False):
        # admin settings
        self.game = game
        self.side = game.player_now
        self.playing_human = playing_human
        self.name = 'Q_Learning_TicTacToe'
        self.policy_file_name = os.path.join(agent_folder, self.name + '_policy')

        # RL agent settings
        self.discount_factor = DISCOUNT_FACTOR
        self.learning_rate = LEARNING_RATE
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.state_values = {}

    def statehash(self, state):
        flat_board = state.reshape(len(game.board[0]) * len(game.board[0]))
        flat_board = list(map(int, flat_board))
        flat_state = str(flat_board)
        return flat_state

    def get_state_values(self, statehash):
        try:
            value = self.state_values[next_board_hash]
        except:
            value = 0
        return value

    def get_available_actions(self):
        i, j = np.where(self.game.board == 0)
        self.avail_actions = list(zip(i,j))

    def get_next_board(self, action):
        next_board = self.game.board.copy()
        next_board[action] = self.side
        return next_board

    def get_best_action(self):
        best_value = -np.inf
        best_action = None
        for action in self.avail_actions:
            next_board = self.get_next_board(action)
            next_board_hash = self.statehash(next_board)
            value = self.get_state_values(next_board_hash)
            if value > best_value:
                best_value = value
                best_action = action
                next_board = next_board
        return best_action, next_board

    def choose_action(self):
        if not self.playing_human:
            if random.random() <= self.epsilon:
                # make a random move
                action = random.choice(self.avail_actions)
                next_state = self.get_next_board(action)
            else:
                action, next_state = self.get_best_action()
        else:
            action, next_state = self.get_best_action()
        return action

    def get_reward(self):
        reward = 0
        win = self.game.check_win()
        tie = self.game.check_tie()
        if win:
            reward = 3
        elif tie:
            reward = 1
        return reward

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def update_values(self, current_state):
        current_hash = self.statehash(current_state)
        _, best_next_state = self.get_best_action()
        next_hash = self.statehash(best_next_state)
        current_Q_value = self.get_state_values(current_hash)
        next_Q_value = self.get_state_values(next_hash)
        if current_Q_value is None:
            current_Q_value = 0
        if next_Q_value is None:
            next_Q_value = 0
        reward = self.get_reward()
        self.state_values[current_hash] = current_Q_value + self.learning_rate * (reward + self.discount_factor * next_Q_value - current_Q_value)

    def do_training(self, epochs):
        for epoch in range(epochs):
            if epoch % 1000 == 0:
                print(f'playing epoch number: {epoch}')
            if epoch % 10 == 0:
                self.decay_epsilon()
            playing = True
            self.game = TicTacToe()
            while playing:
                self.side = self.game.player_now
                self.get_available_actions()
                action = self.choose_action()
                self.game.get_CPU_mark(*action)
                reward = self.get_reward()
                self.update_values(self.game.board)
                win = self.game.check_win()
                tie = self.game.check_tie()
                if win or tie:
                    playing = False
                else:
                    self.game.change_player()

    def save_policy(self):
        with open(self.policy_file_name, 'wb') as f:
            pickle.dump(self.state_values, f)

    def load_policy(self):
        with open(self.policy_file_name, 'rb') as f:
            self.state_values = pickle.load(f)
        

if __name__ == '__main__':
    game = TicTacToe()
    agent = Q_Learning_Agent(game, side = 1, playing_human = False)
    agent.do_training(epochs = 5000)
    agent.save_policy()

        


        

