import numpy as np
from classes.Agent.Agent import Agent

class QLearningAgent(Agent):

    # Constructor of the QLearningAgent class
    # <Params name="name" type="string">The name of the agent</Params>
    # <Params name="epsilon" type="float">Initial value of epsilon. Epsilon corresponds to the probability of the agent to explore rather than to exploit.</Params>
    # <Params name="epsilon_decrease" type="float">the decrease of espilon for each game action taken (Soustraction Operator)</Params>
    # <Params name="epsilon_min" type="float">The minimal value that espilon can reach</Params>
    # <Params name="gamma" type="float">The gamma parameter in the Bellman's equation. It correpsonds to the importance of the next state</Params>
    # <Params name="learning_rate" type="float">The learning rate at which updates will be multiplied</Params>
    def __init__(self, name, epsilon=1., epsilon_decrease=0.1, epsilon_min=0.1, gamma=0.9, learning_rate=0.1):
        super(QLearningAgent, self).__init__(name)
        # parameter checking
        if epsilon < 0 or epsilon > 1:
            raise Exception("epsilon should be between 0 and 1")
        self.epsilon = epsilon 
        if epsilon_decrease < 0:
            raise Exception("epsilon_decrease should be positive")
        self.epsilon_decrease = epsilon_decrease
        if epsilon_min < 0:
            raise Exception("epsilon_min should be positive")
        self.epsilon_min = epsilon_min
        if learning_rate < 0:
            raise Exception("learning_rate should be positive")
        self.learning_rate = learning_rate
        if gamma < 0:
            raise Exception("gamma should be positive")
        self.gamma = gamma
        # instanciate an np array with all possibilities
        # 3's corresponding to the number of value a cell can have (0 = void, 1 = player1, 2 = player2)
        # 9 corresponding to the number of possibilities to play at each state
        self.states = np.zeros((3,3,3,3,3,3,3,3,3,9), dtype=np.float)
        # we need to remember the last state to update it
        self.last_state = None
    
    # Method which choose an action with Q matrix update
    # <Params name="state" type="np.array(np.int)">The current state of the game</Params>
    # <Returns type="np.int">The action choosen by the agent : Reinforcement Learning : Q-Learning</Returns>
    def step_train(self, state):
        action = 0
        if np.random.rand() <= self.epsilon:
            # exploration phase : random choice, on available and not available cells
            # if not available : it will helps the QLearner to make mistakes and be punished for it in order not to repeat them
            action = np.random.choice(9)
        else:
            # converting to tuple helps accessing multidimensionnal arrays easily
            state = tuple(state)
            # exploitation phase : Bellman's equation tells us to select the state which gives us the best propagated reward
            # this will increase mainly the winning known choices
            action = np.argmax(self.states[state])
        temp = tuple([i for i in state] + [action])
        # update weights from previous state
        self.update_from_previous(temp)
        # update the last state
        self.last_state = temp
        return action

    # Method which choose an action without updating Q matrix
    # <Params name="state" type="np.array(np.int)">The current state of the game</Params>
    # <Returns type="np.int">The action choosen by the agent : Reinforcement Learning : Q-Learning</Returns>
    def step(self, state):
        action = 0
        if np.random.rand() <= self.epsilon:
            # exploration phase : random choice, on available and not available cells
            # if not available : it will helps the QLearner to make mistakes and be punished for it in order not to repeat them
            action = np.random.choice(9)
        else:
            # converting to tuple helps accessing multidimensionnal arrays easily
            state = tuple(state)
            # exploitation phase : Bellman's equation tells us to select the state which gives us the best propagated reward
            # this will increase mainly the winning known choices
            action = np.argmax(self.states[state])
        return action
    
    # Method which update the player on a winning game
    # <Params name="reward" type="np.float">The reward amount for winning</Params>
    def update(self, reward):
        # add the reward earned
        if self.last_state != None:
            self.states[self.last_state] = reward
        # epsilon decreasing or minimum checking
        self.epsilon = self.epsilon_min if self.epsilon < self.epsilon_min else self.epsilon - self.epsilon_decrease
    
    def update_from_previous(self, state):
        # converting to tuple helps accessing multidimensionnal arrays easily
        if self.last_state != None:
            # Bellman's Equation : learning_rate * (reward=0. + gamma * current_action - last_action)
            self.states[self.last_state] += self.learning_rate * (self.gamma * self.states[state] - self.states[self.last_state])
            
    def __repr__(self):
        return "<Object Agent:QLearning, Name:{}>".format(self.name)