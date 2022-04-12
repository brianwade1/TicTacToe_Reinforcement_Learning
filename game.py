import random
import numpy as np

class TicTacToe():
    def __init__(self):
        self.board = np.zeros((3,3))
        self.players = {1: 'X', 2: 'O'}
        self.player_now = random.choice(list(self.players.keys()))
        self.playing = True
    
    def make_mark(self, player):
        bad_entry = True
        while bad_entry:
            user_entry = input(f'Player {self.players[self.player_now]} please make a move (format = row #, col #): ')
            try:
                row, col = list(map(int, user_entry.split(',')))
                bad_entry = False
            except:
                print('invalid input')

        if row > 2 or row < 0 or col > 2 or col < 0:
            print('invalid input')
            self.make_mark(player)
        if self.board[row, col] == 0:
            self.board[row, col] = self.player_now
        else:
            print('************')
            print('invalid move')
            self.make_mark(player)

    def get_CPU_mark(self, row, col):
        self.board[row, col] = self.player_now

    def check_win(self):
        # rows
        for row in self.board:
            win = np.all(row == self.player_now)
            if win:
                return win

        # columns 
        for column in self.board.T:
            win = np.all(column == self.player_now)
            if win:
                return win

        # main diagonal
        win = True
        for i in range(3):
            elem = int(self.board[i, i])
            if elem != self.player_now:
                win = False
                break
        if win:
            return win

        # Off Diag
        win = True
        for i in range(3):
            elem = int(self.board[2-i, i])
            if elem != self.player_now:
                win = False
                break
        if win:
            return win
        # all not true so not win
        return False

    def check_tie(self):
        tie = not np.any(self.board == 0)
        return tie
        
    # if winner, stop game
    def win_game(self, player):
        self.playing = False
        self.winner = player
        print('**************************')
        print(f'Player {self.players[player]} wins!!!!')
        print('**************************')
        self.print_board()

    def tie_game(self):
        self.playing = False
        self.winner = None
        print('**************************')
        print(f'Tie Game')
        print('**************************')
        self.print_board()

    def change_player(self):
        if self.player_now == 1:
            self.player_now = 2
        else:
            self.player_now = 1
    
    def print_board(self):
        print('')
        for i, row in enumerate(self.board):
            row_set = []
            for j, col in enumerate(row):
                num = self.board[i, j]
                if num != 0:
                    row_set.append(f' {self.players[num]} ')
                else:
                    row_set.append('   ')

            row_string = row_set[0] + ' | ' + row_set[1] + ' | ' + row_set[2]
            print(row_string)
            if i < 2:
                print('-----------------')
        print('')

    def run(self):
        while self.playing:
            self.print_board()
            self.make_mark(self.player_now)
            win = self.check_win()
            tie = self.check_tie()
            if win:
                self.win_game(self.player_now)
            elif tie:
                self.tie_game()
            else:
                self.change_player()

if __name__ == '__main__':
    game = TicTacToe()
    game.run()





        

