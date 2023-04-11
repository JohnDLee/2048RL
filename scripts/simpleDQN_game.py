# File: simplemlp_game.py
# File Created: Friday, 10th March 2023 10:16:19 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Monday, 10th April 2023 8:59:37 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Plays a game using a trained SimpleMLP model

from pathlib import Path
import sys
import torch
import time
from src.agent_interaction import ENV2048

sys.append('..')
from RL.training_simple import SimpleDQN


if __name__ == '__main__':
    
    # dqn path
    dqn = Path("trained_models/simpleDQN")
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = SimpleDQN(16, 4, None)
    model.load_state_dict(torch.load(str(dqn / 'model_weights.pt')))
    model = model.to(device)
    
    # game env
    env = ENV2048()
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        while True:
            time.sleep(.1)
            # get action
            action = model(state).argmax()
            state, reward, terminated, info = env.step(action.item())
            
            if terminated:
                # loop forever untill killed
                while True:
                    pass 
            else:
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            
