# File: random_game.py
# File Created: Thursday, 9th March 2023 5:29:46 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Sunday, 2nd April 2023 5:36:36 am
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Plays a completely random game of 2048

from src.agent_interaction import ENV2048
import time
import random

if __name__ == '__main__':
    
    env = ENV2048()
    
    for i in range(10):
        done = False
        env.reset()
        while not done:
            time.sleep(.01)
            action = env.random_action()
            _, _, done, _ = env.step(action)
    
    while True:
        pass
        
        
        