from src.agent_interaction import ENV2048
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import copy
import random
import math



        

class SimpleDQN(nn.Module):
    
    def __init__(self, input_neurons, output_neurons, hidden_layers):
        """create a simple MLP DQN

        Args:
            input_neurons (int): number of input neurons
            output_neurons (int): number of output results
            hidden_layers (list): list of numbers describing # of hidden layers and their sizes
        """
        super().__init__()
        
        # activations
        self.relu = nn.ReLU()
        self.softmax = nn.SoftMax()
        
        # model
        self.model = nn.Sequential([nn.Linear(input_neurons, 128),
                                    self.relu,
                                   nn.Linear(128, 128),
                                   self.relu,
                                   nn.Linear(128, output_neurons)]) #! Fill this out...
        
        
    def forward(self, x):
        return self.model(x.flatten())

class TrainDQN():
    
    def __init__(self, model, optimizer, loss_fn, game_env, device, **kwargs):
        
        ## HYPERPARAMS
        self.BATCH_SIZE = kwargs['BATCH_SIZE'] if 'BATCH_SIZE' in kwargs else 128 # batchsize for each training step
        self.GAMMA = kwargs['GAMMA'] if 'GAMMA' in kwargs else 0.99 # discount rate for rewares
        self.EPS_START = kwargs['EPS_START'] if 'EPS_START' in kwargs else 0.9 # prob of choosing a random action
        self.EPS_END = kwargs['EPS_END'] if 'EPS_END' in kwargs else 0.05 # end prob of choosing a random action
        self.EPS_DECAY = kwargs['EPS_DECAY'] if 'EPS_DECAY' in kwargs else 1000 # how many actions before decaying
        self.TAU = kwargs['TAU'] if 'TAU' in kwargs else 0.005 # update rate of target network

        ## MODELS
        self.policy_model = model.to(device)
        self.target_model = copy.deepcopy(model).to(device)
        self.device = device

        ## OPTIMIZER
        self.optimizer = optimizer
        
        ## LOSS FN
        self.loss_fn = loss_fn
        
        ## MEMORY
        self.memory = ReplayMemory(kwargs['BUF_SIZE'] if 'BUF_SIZE' in kwargs else 100000)
        
        ## GAME ENV
        self.game_env = game_env
        
        ## INTERNALS
        self.steps_completed = 0
    
    def select_action(self, state):
        # threshold computed based on decay
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_completed / self.EPS_DECAY)
        self.steps_completed += 1
        
        # if you choose according to policy
        if random.random() > eps_threshold:
            with torch.no_grad():
                
                #! Not sure if this is correct.
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_model(state).max(1)[1].view(1, 1)
        # if you choose random
        else:
            return torch.tensor([[self.game_env.random_action()]], device=self.device, dtype=torch.long)


        
        
class ReplayMemory():
    """Store previous actions to replay them.
    """
    
    def __init__(self, buf_size = 60000):
        self.memory = deque([], maxlen = buf_size)
    
    def push(self):
        ''' save a transition '''
        pass
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
        
        
        
def main():
    
    ## HYPERPARAMS (most are defaulted already)
    LR = 1e-4 # LR of optimizer
        
    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## GAME ENVIRONMENT
    game_env = ENV2048()
    
    ## GAME OPTIONS/STATE
    # get number of actions and state size
    num_actions = game_env.num_actions()
    num_obs = game_env.get_state().flatten()
    
    ## NETWORK
    model = SimpleDQN(num_obs, num_actions, None) # runs policy & updates each step


    ## OPTIMIZER/MEMORY/LOSS
    optimizer = optim.AdamW(model.parameters(),
                            lr=LR,
                            amsgrad=True)
    loss_fn = nn.HuberLoss()
    
    ## TRAINER
    # use default hyperparams
    trainer = TrainDQN(model=model,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       game_env=game_env) 
    
    
    
    

    
    
    
    


if __name__ == '__main__':
    main()
