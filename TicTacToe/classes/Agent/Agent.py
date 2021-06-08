import numpy as np

class Agent(object):

    # Constructor of the Agent class
    # <Params name="name" type="string">The name of the agent</Params>
    def __init__(self, name):
        super(Agent, self).__init__()
        self.name = name
        self.game_played=0
        self.results=np.array([0,0,0,0,0], dtype=np.int32)
    
    # Method which choose an action
    # <Params name="state" type="np.array(np.int)">The current state of the game</Params>
    # <Returns type="np.int">The action choosen by the agent : random on available</Returns>
    def step(self, state):
        action = np.random.choice([i for i in range(state.shape[0]) if state[i] == 0])
        return action
    
    # Method which update the player on a winning game
    # <Params name="reward" type="np.float">The reward amount for winning</Params>
    # <Params name="result" type="string">The result of the game</Params>
    def update(self, reward, result):
        self.game_played += 1
        self.results[result] += 1
        pass
    
    def __str__(self):
        return self.name
    def __repr__(self):
        return "<Object Agent:Random, Name:{}>".format(self.name)