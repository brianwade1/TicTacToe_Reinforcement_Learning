import random

class TicTacToe():
    def __init__(self):
        self.setup_str = '-'
        self.board = [[self.setup_str,self.setup_str,self.setup_str], 
                        [self.setup_str,self.setup_str,self.setup_str],
                        [self.setup_str,self.setup_str,self.setup_str]]
        self.players = ['X', 'O']
        #self.player_now = random.choice(['X', 'O'])
        self.player_now = random.choice(self.players)
        self.playing = True
    
    def make_mark(self, player):
        bad_entry = True
        while bad_entry:
            user_entry = input(f'Player {self.player_now} please make a move (format = #,#): ')
            try:
                row, col = list(map(int, user_entry.split(',')))
                bad_entry = False
            except:
                print('invalid input')

        if row > 2 or row < 0 or col > 2 or col < 0:
            print('invalid input')
            self.make_mark(player)
        if self.board[row][col] == '-':
            self.board[row][col] = self.player_now
        else:
            print('************')
            print('invalid move')
            self.make_mark(player)

    def check_win(self, player):
        # rows
        for row in self.board:
            win = True
            for elem in row:
                if elem != self.player_now:
                    win = False
                    break
            if win:
                return win

        # columns 
        for column in range(3):
            win = True
            for row in range(3):
                elem = self.board[row][column]
                if elem != self.player_now:
                    win = False
                    break
            if win:
                return win

        # main diagonal
        win = True
        for i in range(3):
            elem = self.board[i][i]
            if elem != self.player_now:
                win = False
                break
        if win:
            return win

        # Off Diag
        win = True
        for i in range(3):
            elem = self.board[2-i][2-i]
            if elem != self.player_now:
                win = False
                break
        if win:
            return win
        # all not true so not win
        return False

    def check_tie_game(self):
        tie = True
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == self.setup_str:
                    tie = False
                    return tie
        return tie
        
    # if winner, stop game
    def win_game(self, player):
        self.playing = False
        print('**************************')
        print(f'Player {player} wins!!!!')
        print('**************************')
        self.print_board()

    def tie_game(self):
        self.playing = False
        print('**************************')
        print(f'Tie Game')
        print('**************************')
        self.print_board()

    def change_player(self):
        if self.player_now == 'X':
            self.player_now = 'O'
        else:
            self.player_now = 'X'
    
    def print_board(self):
        for row in self.board:
            print(row)
        print('-----------------')

    def run(self):
        while self.playing:
            self.print_board()
            self.make_mark(self.player_now)
            win = self.check_win(self.player_now)
            tie = self.check_tie_game()
            if win:
                self.win_game(self.player_now)
            elif tie:
                self.tie_game()
            else:
                self.change_player()

if __name__ == '__main__':
    game = TicTacToe()
    game.run()





        

