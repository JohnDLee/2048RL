# File: training_simple.py
# File Created: Friday, 10th March 2023 10:24:44 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Saturday, 1st April 2023 4:28:55 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Training script for a simple DQN, repurposed from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


from src.agent_interaction import ENV2048
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from itertools import count
import numpy as np
import copy
from pathlib import Path
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
        self.softmax = nn.Softmax()
        
        # model
        self.model = nn.Sequential(nn.Linear(input_neurons, 1024),
                                    self.relu,
                                   nn.Linear(1024, 512),
                                   self.relu,
                                   nn.Linear(512, 256),
                                   self.relu,
                                   nn.Linear(256, output_neurons)) #! Fill this out...
        
        
    def forward(self, x):
        return self.model(x.flatten(1).log2().nan_to_num(neginf=0))

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
        
        ## TRAIN INTERNALS
        self.steps_completed = 0
        self.episode_durations = []
        self.ax = None
        self.fig = None
    
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
                return self.policy_model(state).argmax().view(1,1)
        # if you choose random
        else:
            return torch.tensor([[self.game_env.random_action()]], device=self.device, dtype=torch.long)

    def optimize_model_step(self):
        
        ## CHECK MEM LEN
        # do nothing if memory is not long enough yet.
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        ## GET BATCH
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose batch of transitions into a single transition with batches 
        batch = Transition(*zip(*transitions))
        
        ## SPLIT INTO TENSORS
        
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss

        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
        self.optimizer.step()
            
    def optimize_model(self, num_episodes):
    
        for i_episode in range(num_episodes):
            # Initialize the environment and get it's state
            state = self.game_env.reset()
            # flatten as torch tensor to allow as input.
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            total_score = 0
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, info = self.game_env.step(action.item())
                total_score += reward
                reward = torch.tensor([reward], device=self.device)
                

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model_step()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                if self.steps_completed % 1000:
                    target_model_state_dict = self.target_model.state_dict()
                    policy_model_state_dict = self.policy_model.state_dict()
                    for key in policy_model_state_dict:
                        target_model_state_dict[key] = policy_model_state_dict[key]*self.TAU + target_model_state_dict[key]*(1-self.TAU)
                    self.target_model.load_state_dict(target_model_state_dict)

                if terminated:
                    self.episode_durations.append(total_score)
                    self.fig, self.ax = self.plot_durations(self.fig, self.ax, False)
                    break
        
    def plot_durations(self, fig=None, ax: plt.Axes=None, show_result=False):
        if not fig:
            fig = plt.figure(1)
        if not ax:
            ax = fig.add_subplot(111)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            ax.set_title('Result')
        else:
            ax.clear()
            ax.set_title('Training...')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            ax.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        fig.show()
        return fig, ax
        
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory():
    """Store previous actions to replay them.
    """
    
    def __init__(self, buf_size = 60000):
        self.memory = deque([], maxlen = buf_size)
    
    def push(self, *args):
        ''' save a transition '''
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    


        
        
def main():
        
    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## GAME ENVIRONMENT
    game_env = ENV2048()
    
    ## GAME OPTIONS/STATE
    # get number of actions and state size
    num_actions = game_env.num_actions()
    num_obs = len(game_env.get_state().flatten())
    
    ## NETWORK
    model = SimpleDQN(num_obs, num_actions, None) # runs policy & updates each step


    ## OPTIMIZER/MEMORY/LOSS
    LR = 1e-4 # LR of optimizer
    optimizer = optim.AdamW(model.parameters(),
                            lr=LR,
                            amsgrad=True)
    loss_fn = nn.SmoothL1Loss()
    
    ## TRAINER
    # use default hyperparams
    trainer = TrainDQN(model=model,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       game_env=game_env,
                       device=device,) 
    
    
    trainer.optimize_model(500)
    
    trainer.plot_durations(trainer.fig, trainer.ax, show_result=True)
    
    # Save model
    p = Path('trained_models/simpleDQN')
    p.mkdir(exist_ok = True, parents = True)
    
    torch.save(model.state_dict(), str(p / 'model_weights.pt'))
    trainer.fig.savefig(str(p/'train_results.png'))


if __name__ == '__main__':
    main()
