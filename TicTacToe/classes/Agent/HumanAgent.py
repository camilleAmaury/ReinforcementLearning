from classes.Agent.Agent import Agent

class HumanAgent(Agent):

    # Constructor of the HumanAgent class
    # <Params name="name" type="string">The name of the agent</Params>
    def __init__(self, name):
        super(HumanAgent, self).__init__(name)
        
    # Method which choose an action
    # <Params name="state" type="np.array(np.int)">The current state of the game</Params>
    # <Returns type="np.int">The action choosen by the agent : Player asked</Returns>
    def step(self, state):
        action = int(input("Choose an action \n (for example '0' corresponds to the first column, first line cell)\n Available : {} = ".format(
            [i for i in range(state.shape[0]) if state[i] == 0])))
        return action
    
    def __repr__(self):
        return "<Object Agent:HumanAgent, Name:{}>".format(self.name)