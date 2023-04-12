# File: simplemlp_game.py
# File Created: Friday, 10th March 2023 10:16:19 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Wednesday, 12th April 2023 12:37:05 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Plays a game using a trained SimpleMLP model

from pathlib import Path
import sys
import torch
import time
from src.agent_interaction import ENV2048

sys.path.append('..')
from RL.training_simple import SimpleDQN


if __name__ == '__main__':
    
    # dqn path
    dqn = Path("trained_models/simpleDQN500")
    
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
        random = False
        while True:
            time.sleep(.1)
            # get action
            
            if random:
                # Random move if previous was an invalid move
                state, reward, terminated, info = env.step(env.random_action())
                random = False
            else:
                # Use best move
                scores = model(state)
                action = scores.argmax()
                state, reward, terminated, info = env.step(action.item())
            
            
            if info == env.game.INVALID_MOVE:
                random = True
            
            if terminated:
                # loop forever untill killed
                env.game.game_over()
                while True:
                    pass 
            else:
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            
