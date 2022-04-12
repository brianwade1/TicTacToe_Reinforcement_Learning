# Standard Libraries
import random, os, pickle, time
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
        self.win_reward = WIN_REWARD
        self.tie_reward = TIE_REWARD
        if playing_human:
            self.side = side
        else:
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
        self.state_values = {1: {}, 2: {}}

    def statehash(self, state):
        flat_board = state.reshape(len(self.game.board[0]) * len(self.game.board[0]))
        flat_board = list(map(int, flat_board))
        flat_state = str(flat_board)
        return flat_state

    def reverse_statehash(self, statehash):
        str_list = statehash.strip('][').split(', ')
        num_list = [int(x) for x in str_list]
        state_board_flat = np.asarray(num_list)
        state_board = np.reshape(state_board_flat, (len(self.game.board[0]), len(self.game.board[0])))
        return state_board

    def get_state_values(self, side, statehash):
        try:
            value = self.state_values[side][statehash]
        except:
            value = 0
        return value

    def get_available_actions(self, board):
        i, j = np.where(board == 0)
        self.avail_actions = list(zip(i,j))

    def get_next_board(self, board, action):
        next_board = board.copy()
        next_board[action] = self.side
        return next_board

    def get_best_action(self, board):
        best_value = -np.inf
        best_action = None
        self.get_available_actions(board)
        random.shuffle(self.avail_actions)
        for action in self.avail_actions:
            next_board = self.get_next_board(board, action)
            next_board_hash = self.statehash(next_board)
            value = self.get_state_values(self.side, next_board_hash)
            if value > best_value:
                best_value = value
                best_action = action
                best_board = next_board
        return best_action, best_board, best_value

    def choose_action(self):
        if self.playing_human:
            action, next_state, _ = self.get_best_action(self.game.board)
            return action
        if not self.playing_human:
            if random.random() <= self.epsilon:
                action = random.choice(self.avail_actions) #epsilon greedy
            else:
                action, next_state, _ = self.get_best_action(self.game.board)
            return action

    def get_reward(self):
        reward = 0
        win = self.game.check_win()
        tie = self.game.check_tie()
        if win:
            reward = self.win_reward
        elif tie:
            reward = self.tie_reward
        return reward, win, tie

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def get_other_side(self):
        for x in list(self.game.players.keys()):
                if x != self.side:
                    other_side = x
        return other_side

    def update_values(self, current_state, reward):
        #current_hash = self.statehash(current_state)
        if reward == self.win_reward: # current side won so gets reward, other side gets negative win reward
            # winning side
            current_hash = self.state_hist[self.side][-1]
            current_Q_value = self.get_state_values(self.side, current_hash)
            self.state_values[self.side][current_hash] = current_Q_value + self.learning_rate * (reward - current_Q_value)
            #loosing side
            other_side = self.get_other_side()
            last_hash = self.state_hist[other_side][-1] #board right before other side won
            last_Q_value = self.get_state_values(other_side, current_hash)
            self.state_values[other_side][last_hash] = last_Q_value + self.learning_rate * ((-reward) - last_Q_value)
        elif reward == self.tie_reward: #tie game - both sides get tie reward
            #side that just played
            current_hash = self.state_hist[self.side][-1]
            current_Q_value = self.get_state_values(self.side, current_hash)
            self.state_values[self.side][current_hash] = current_Q_value + self.learning_rate * (reward - current_Q_value)
            # side that played just one turn ago
            other_side = self.get_other_side()
            last_hash = self.state_hist[other_side][-1] #board right before other side caused a tie
            last_Q_value = self.get_state_values(other_side, last_hash)
            self.state_values[other_side][last_hash] = last_Q_value + self.learning_rate * (reward - last_Q_value)
        else:
            if len(self.state_hist[self.side]) > 1:
                # side that just played - the board before the other player went
                last_played_hash = self.state_hist[self.side][-2]
                current_Q_value = self.get_state_values(self.side, last_played_hash)

                # Get the board before my current move... when the opponent just moved
                other_side = self.get_other_side()
                last_board = self.reverse_statehash(self.state_hist[other_side][-1])
                _, best_next_state, next_Q_value = self.get_best_action(last_board)
                next_hash = self.statehash(best_next_state)
                self.state_values[self.side][last_played_hash] = current_Q_value + self.learning_rate * (self.discount_factor * next_Q_value - current_Q_value)
            

    def do_training(self, games_to_play):
        # Check for existing policy to continue training
        if os.path.isfile(self.policy_file_name):
            self.load_policy()
        # Train!
        for current_game in range(games_to_play):
            if current_game % 1000 == 0:
                print(f'playing game number: {current_game}')
            if current_game % 10000 == 0:
                self.decay_epsilon()
            playing = True
            self.game = TicTacToe()
            self.state_hist = {1: [], 2: []}
            while playing:
                self.side = self.game.player_now
                self.get_available_actions(self.game.board)
                action = self.choose_action()
                self.game.get_CPU_mark(*action)
                self.state_hist[self.side].append(self.statehash(self.game.board))
                reward, win, tie = self.get_reward()
                self.update_values(self.game.board, reward)
                if win or tie:
                    playing = False
                else:
                    self.game.change_player()
        self.save_policy()

    def play_human(self, game):
        # Check for existing policy to continue training
        if os.path.isfile(self.policy_file_name):
            self.load_policy()
            self.game = game
        else:
            print('No policy available!')
            return
        if self.side == self.game.player_now:
            order = 'first'
        else: 
            order = 'second'
        print(f'CPU is the {self.game.players[self.side]} mark and will be going {order}')
        time.sleep(0.5)
        hist_board = {1: [], 2: []} # save history for later learning (imitation learning)
        hist_side = []
        hist_reward = [] 
        while self.game.playing:
            if self.side == self.game.player_now:
                self.game.print_board()
                self.get_available_actions(self.game.board)
                action = self.choose_action()
                print(f'CPU chooses action {action}')
                time.sleep(1)
                self.game.get_CPU_mark(*action)
            else:
                self.game.print_board()
                self.game.make_mark(self.game.player_now)
            hist_side.append(self.game.player_now)
            hist_board[self.game.player_now].append(self.statehash(self.game.board))
            reward, win, tie = self.get_reward()
            hist_reward.append(reward)
            if win:
                self.game.win_game(self.game.player_now)
            elif tie:
                self.game.tie_game()
            else:
                self.game.change_player()
        # Imitation learning to update policy
        self.state_hist = {1: [], 2: []}
        side_count = {1: 0, 2: 0}
        for i, side in enumerate(hist_side):
            self.side = side
            replay_hash = hist_board[side][side_count[side]]
            replay_board = self.reverse_statehash(replay_hash)
            reward = hist_reward[i]
            self.state_hist[side].append(replay_hash)
            self.update_values(replay_board, reward)
            side_count[side] += 1
        self.save_policy()

    def save_policy(self):
        with open(self.policy_file_name, 'wb') as f:
            pickle.dump(self.state_values, f)

    def load_policy(self):
        with open(self.policy_file_name, 'rb') as f:
            self.state_values = pickle.load(f)
        

if __name__ == '__main__':
    game = TicTacToe()
    agent = Q_Learning_Agent(game, side = 1, playing_human = False)
    agent.do_training(games_to_play = 500000)
    

        


        

