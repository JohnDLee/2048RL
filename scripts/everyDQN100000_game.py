# File: simplemlp_game.py
# File Created: Friday, 10th March 2023 10:16:19 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Sunday, 16th April 2023 3:22:48 pm
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
    dqn = Path("trained_models/everyDQN100000")
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = SimpleDQN(16, 4, None)
    model.load_state_dict(torch.load(str(dqn / 'model_weights.pt'), map_location=device))
    model = model.to(device)
    
    # game env
    env = ENV2048(gui = True)
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        next_choice = 0
        while True:
            time.sleep(.1)
            # get action

            # Use best move
            scores = model(state)
            action = scores.topk(4)[1][0][next_choice]
            state, reward, terminated, info = env.gui_step(action.item())
            
            
            if info == env.game.INVALID_MOVE:
                next_choice += 1
            else:
                next_choice = 0
            
            if terminated:
                # loop forever untill killed
                env.game.game_over()
                while True:
                    pass 
            else:
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            
