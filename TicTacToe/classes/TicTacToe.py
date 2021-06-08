import numpy as np
from tqdm import tqdm

class TicTacToe(object):
    
    # Constructor of the TicTacToe class
    # <Params name="player1" type="Agent">The first player</Params>
    # <Params name="player2" type="Agent">The second player</Params>
    # <Params name="rewards" type="object">
    #     <Contains key="win_reward" type="float">The amount of reward for a win</Contains>
    #     <Contains key="lose_reward" type="float">The amount of reward for a defeat</Contains>
    #     <Contains key="draw_reward" type="float">The amount of reward for a draw</Contains>
    #     <Contains key="error_reward" type="float">The amount of reward to make an already done move</Contains>
    #     <Contains key="null_reward" type="float">The amount of reward for another agent error</Contains>
    # </Params>
    def __init__(self, player1, player2, rewards):
        super(TicTacToe, self).__init__()
        # Initial board state, {0 = void, 1 = player 1, 2 = player 2}
        self.board = np.array([0,0,0,0,0,0,0,0,0], dtype=int)
        self.players = np.array([player1, player2])
        np.random.shuffle(self.players)
        self.markers = {"0":"v", "1":"x", "2":"o"}
        self.rewards = rewards

    # Method which re-instanciate a new game
    def reset_env(self):
        self.board = np.array([0,0,0,0,0,0,0,0,0])
        np.random.shuffle(self.players)
        
    # Method launch a game
    # <Params name="verbose" type="int">The more verbose is high, the more logs you will have</Params>
    def run_game(self, verbose=0):
        done = False
        for i in range(self.board.shape[0]):
            turn = i%2
            agent_turn = self.players[turn]
            agent_value = turn+1
            agent_not_turn = self.players[agent_value%2]
            # agent take action
            action = agent_turn.step(self.board)
            if self.board[action] == 0:
                # correct action
                self.board[action] = agent_value
                if verbose==2:
                    print(self)
                # checks for victory
                if not self.has_win(agent_value):
                    if self.is_board_full(verbose):
                        agent_turn.update(self.rewards["draw_reward"], 1)
                        agent_not_turn.update(self.rewards["draw_reward"], 1)
                        break
                else:
                    agent_turn.update(self.rewards["win_reward"], 0)
                    agent_not_turn.update(self.rewards["lose_reward"], 2)
                    if verbose >= 1:
                        print("Player {} wins :\n{}".format(agent_turn, self))
                    break
            else:
                # the player choose a wrong action (already taken)
                if verbose >= 1:
                    print("Player {} choosed a wrong action : end".format(agent_turn))
                agent_turn.update(self.rewards["error_reward"], 3)
                agent_not_turn.update(self.rewards["null_reward"], 4)
                break
            
    # Method used to run multiples games and train RL agents
    # <Params name="epochs" type="int">The number of game to play</Params>
    # <Params name="verbose" type="int">The more verbose is high, the more logs you will have</Params>
    def train(self, epochs=1, verbose=0):
        for _ in tqdm(range(epochs), miniters=10000):
            self.run_game(verbose)
            self.reset_env()
            
    # Method used to check whether the board is full or not
    # <Params name="verbose" type="int">The more verbose is high, the more logs you will have</Params>
    def is_board_full(self, verbose):
        cond = self.board[self.board == 0].shape[0] == 0
        if cond and verbose >= 1:
            print("Game is finished with no winner")
        return cond
    
    # Method which check if a user win
    # <Params name="marker" type="string">The marker representing the agent</Params>
    def has_win(self, marker):
        cond = False
        # row
        cond = cond or (self.board[0] == marker and self.board[1] == marker and self.board[2] == marker) or \
            (self.board[3] == marker and self.board[4] == marker and self.board[5] == marker) or \
            (self.board[6] == marker and self.board[7] == marker and self.board[8] == marker)
        # column
        cond = cond or (self.board[0] == marker and self.board[3] == marker and self.board[6] == marker) or \
            (self.board[1] == marker and self.board[4] == marker and self.board[7] == marker) or \
            (self.board[2] == marker and self.board[5] == marker and self.board[8] == marker)
        # diagonals
        cond = cond or (self.board[0] == marker and self.board[4] == marker and self.board[8] == marker) or \
            (self.board[2] == marker and self.board[4] == marker and self.board[6] == marker)
        return cond
   
    def __str__(self):
        s = "_____________\n"
        for i in range(self.board.shape[0]):
            if i % 3 == 0 and i != 0:
                s += "|\n_____________\n"
            s += "| {} ".format(self.markers[str(self.board[i])])
        s += "|\n_____________"
        return s
    
    def __repr__(self):
        return "<Object TicTacToe, Agents:[{},{}]>".format(self.players[0], self.players[1])
    
   
        
            
        