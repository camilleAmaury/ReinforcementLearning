import numpy as np
from classes.Agent.Agent import Agent
from keras.models import load_model

class DQLearningAgent(Agent):

    # Constructor of the DQLearningAgent class
    # <Params name="name" type="string">The name of the agent</Params>
    # <Params name="n_actions" type="int">The number of actions per state</Params>
    # <Params name="fn" type="function">Functions which builds a neural network</Params>
    # <Params name="mem_size" type="int">The size of the replay buffer</Params>
    # <Params name="batch_size" type="int">The size of batch for each training</Params>
    # <Params name="epsilon" type="float">Initial value of epsilon. Epsilon corresponds to the probability of the agent to explore rather than to exploit.</Params>
    # <Params name="epsilon_decrease" type="float">the decrease of espilon for each game action taken (Soustraction Operator)</Params>
    # <Params name="epsilon_min" type="float">The minimal value that espilon can reach</Params>
    # <Params name="gamma" type="float">The gamma parameter in the Bellman's equation. It correpsonds to the importance of the next state</Params>
    # <Params name="learning_rate" type="float">The learning rate at which updates will be multiplied</Params>
    def __init__(self, name,  n_actions, input_dims, fn, mem_size, batch_size, epsilon=1., epsilon_decrease=0.1, epsilon_min=0.1, gamma=0.9, learning_rate=0.1):
        super(DQLearningAgent, self).__init__(name)
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
        if gamma < 0:
            raise Exception("gamma should be positive")
        self.gamma = gamma


        if n_actions <= 0:
            raise Exception("gamma should be positive and superior to 0")
        self.action_space = np.arange(n_actions, dtype=np.int8)
        self.n_actions = n_actions
        if batch_size <= 0:
            raise Exception("batch_size should be positive and superior to 0")
        self.batch_size = batch_size
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, 
                                   discrete=True)
        self.q_eval = fn(learning_rate, n_actions, input_dims, 256, 256)
    
    # Method which store a decision
    # <Params name="state" type="np.array(int)">The state during when the action was taken</Params>
    # <Params name="action" type="int">The action choosen</Params>
    # <Params name="reward" type="float">The reward obtainer by taking this action at specific state</Params>
    # <Params name="new_state" type="np.array(int)">The state following the previous state action pair</Params>
    # <Params name="done" type="int">If the game is done due to pair actions - states</Params>
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    # Method select an action
    # <Params name="state" type="np.array(int)">The current state</Params>
    # <Returns type="int">The action choosen by the agent : Reinforcement Learning : Q-Learning</Returns>
    def choose_action(self, state):
        state = state[np.newaxis,:]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action
    
    # Method which train the network
    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        action_values = self.action_space
        action_indices = np.dot(action, action_values)
        
        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)
        
        q_target = q_eval.copy()
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        q_target[batch_index, action_indices] = reward + \
            self.gamma*np.max(q_next, axis=1)*done
        
        _ = self.q_eval.fit(state, q_target, verbose=0)
        
        self.epsilon = self.epsilon*self.epsilon_decrease if self.epsilon > \
            self.epsilon_min else self.epsilon_min
        
    def save_model(self):
        self.q_eval.save(self.name)
    def load_model(self):
        self.q_eval = load_model(self.name)
    
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



class ReplayBuffer(object):

    # Constructor of the ReplayBuffer class
    # <Params name="max_size" type="int">The number of rows allocated to the replay buffer</Params>
    # <Params name="input_shape" type="int">The size of the input layer</Params>
    # <Params name="n_actions" type="int">The number of actions possible at a each state</Params>
    # <Params name="discrete" type="boolean">If the action selected his continuous or discrete</Params>
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        super(ReplayBuffer, self).__init__()
        self.mem_size = max_size
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.mem_counter = 0
        
    # Method which store a state-action pair and the result of it (reward + next state)
    # <Params name="state" type="np.array(int)">The state during when the action was taken</Params>
    # <Params name="action" type="int">The action choosen</Params>
    # <Params name="reward" type="float">The reward obtainer by taking this action at specific state</Params>
    # <Params name="new_state" type="np.array(int)">The state following the previous state action pair</Params>
    # <Params name="done" type="int">If the game is done due to pair actions - states</Params>
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_counter += 1
        
    # Method which returns a training batch
    # <Params name="batch_size" type="int">The state during when the action was taken</Params>
    # <Returns type="tuple">
    #   <Object type="np.matrix(int)" name="states">The states before taking action</Object>
    #   <Object type="np.array(int)" name="actions">The actions taken</Object>
    #   <Object type="np.array(int)" name="rewards">The rewards obtained with pair actions - states</Object>
    #   <Object type="np.matrix(int)" name="new_states">The next states following pair actions - states</Object>
    #   <Object type="np.array(int)" name="terminal">If the games are done due to this pair actions - states</Object>
    # </Returns>
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]
        
        return states, actions, rewards, new_states, terminal