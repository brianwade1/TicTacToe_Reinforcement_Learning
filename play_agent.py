# Other scripts
from game import TicTacToe
from Q_learning_agent import Q_Learning_Agent
from settings import *

def main():
    game = TicTacToe()
    marker = input('Would you like to be X or O? ')
    if marker == 'X':
        side = 2 #CPU is Os
    else:
        side = 1 #CPU is Xs
    agent = Q_Learning_Agent(game, side = side, playing_human = True)
    agent.play_human(game)


if __name__ == '__main__':
    main()