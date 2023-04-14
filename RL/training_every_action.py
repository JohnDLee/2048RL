# File: training_simple.py
# File Created: Friday, 10th March 2023 10:24:44 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Friday, 14th April 2023 10:39:16 am
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
import argparse



        

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
                                   nn.Linear(256, output_neurons, bias = False)) #! Fill this out...
        
        
    def forward(self, x):
        return self.model(x.flatten(1).log2().nan_to_num(neginf=0))

class TrainDQN():
    
    def __init__(self, model, optimizer, loss_fn, game_env, device, save_dir, **kwargs):
        
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
        self.game_step = game_env.gui_step if game_env.gui else game_env.step
        
        ## save dirs
        self.save_dir = str(save_dir)
        self.save_plot = str(Path(self.save_dir) / 'train_results.pt')
        self.save_weights = str(Path(self.save_dir) / 'model_weights.png')
        
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
                # found, so we pick action with the largest expected reward.
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
        
        # handle multiple elements for each
        non_final_mask = torch.vstack(tuple(map(lambda x: torch.tensor(tuple(map(lambda s: s is not None,
                                            x)), device=self.device, dtype=torch.bool),
                                            batch.next_state)))
        
        non_final_next_states = torch.cat([s for row in batch.next_state for s in row
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.vstack(batch.action)
        reward_batch = torch.vstack(batch.reward)
        
        # print("state:", state_batch)
        # print("action_batch:", action_batch)
        # print("reward_batch:", reward_batch)
        # print("non_final_next_states", non_final_next_states)
  
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # print(self.policy_model(state_batch))
        # print(action_batch)
        state_action_values = self.policy_model(state_batch).gather(1, action_batch)
        # above is the predicted reward 
        #m = torch.where(state_action_values < 0)[0]
        #print("predicted reward", (state_batch[m], action_batch[m], state_action_values[m]))
        
    

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        #! 4 is specifically for actions
        next_state_values = torch.zeros((self.BATCH_SIZE, len(action_batch[0])), device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0]
            
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
       
        #print(state_action_values, expected_state_action_values)

        # Compute Huber loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
        self.optimizer.step()
            
    def optimize_model(self, num_episodes, save_every = 100):
    
        for i_episode in range(num_episodes):
            print(f"Episode {i_episode}/{num_episodes}", flush=True)
            # Initialize the environment and get it's state
            state = self.game_env.reset()
            # flatten as torch tensor to allow as input.
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            for t in count():
                
                # observe every action using a pseudo step
                rewards = []
                next_states = []
                for a_idx in range(self.game_env.num_actions()):
                
                    observation, reward, terminated, info = self.game_env.pseudo_step(a_idx)
                    rewards.append(reward)

                    if terminated:
                        next_state = None
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    next_states.append(next_state)
                # take real step
                action = self.select_action(state)
                observation, reward, terminated, info = self.game_step(action.item())
                    
                # store transition in memory   
                rewards = torch.tensor(rewards, device=self.device)
                
                actions = torch.tensor([range(self.game_env.num_actions())], device=self.device)
                # Store the transition in memory
                self.memory.push(state, actions, next_states, rewards)

                # Move to the next state
                state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0) if not terminated else None

                # Perform one step of the optimization (on the policy network)
                self.optimize_model_step()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_model_state_dict = self.target_model.state_dict()
                policy_model_state_dict = self.policy_model.state_dict()
                for key in policy_model_state_dict:
                    target_model_state_dict[key] = policy_model_state_dict[key]*self.TAU + target_model_state_dict[key]*(1-self.TAU)
                self.target_model.load_state_dict(target_model_state_dict)

                if terminated:
                    self.episode_durations.append(self.game_env.backend.get_score())
                    self.fig, self.ax = self.plot_durations(self.fig, self.ax, False)
                    # save every so often.
                    if i_episode % save_every == 0:
                        self.save_results()
                    break
                
        self.plot_durations(self.fig, self.ax, show_result=True)
        self.save_results()
        
        
    def save_results(self):
        ''' save results '''
        torch.save(self.policy_model.state_dict(), self.save_weights)
        self.fig.savefig(self.save_plot)
        
        
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
    


        
        
def main(args):
        
    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## GAME ENVIRONMENT
    game_env = ENV2048(gui = args.gui)
    
    ## GAME OPTIONS/STATE
    # get number of actions and state size
    num_actions = game_env.num_actions()
    num_obs = len(game_env.get_state().flatten())
    
    ## NETWORK
    model = SimpleDQN(num_obs, num_actions, None) # runs policy & updates each step


    ## OPTIMIZER/MEMORY/LOSS
    LR =2.5e-4 # LR of optimizer
    optimizer = optim.Adam(model.parameters(),
                            lr=LR,
                            )
    loss_fn = nn.HuberLoss()
    
    
    # save dir
    save_dir = Path('trained_models/everyDQN' + str(args.eps))
    save_dir.mkdir(exist_ok = True, parents = True)
    
    ## TRAINER
    # use default hyperparams
    trainer = TrainDQN(model=model,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       game_env=game_env,
                       device=device,
                       save_dir=save_dir) 
    
    trainer.optimize_model(args.eps)
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Training for DQN')
    parser.add_argument("-gui", action = 'store_true', default = False, help = 'Use the GUI to visualize results in real time. GUI will cause slowdowns and should not be used for actual training.')
    parser.add_argument("-eps", type = int, default = 500, help = 'number of episodes to train for')
    
    args = parser.parse_args()
    main(args)
    if args.gui:
        while True:
           pass
