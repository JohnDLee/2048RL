
from src.gui import GAME_2048
import numpy as np

class ENV2048():
    
    def __init__(self):
        # start a game
        self.game = GAME_2048()
        self.actions = [self.game.left, self.game.up, self.game.right, self.game.down]
        
    def num_actions(self):
        ''' gives the total number of actions possible '''
        return len(self.actions)
    
    def random_action(self):
        ''' gives a random action to do '''
        return np.random.randint(4) # 0 - 3
    
    def step(self, action):
        ''' takes an action '''
        state = self.actions[action](None) # get state ccorresponding to action
        
        if state == self.game.SUCCESSFUL_MOVE:
            return (self.game.backend.get_state(), self.game.backend.get_last_reward(), False)
        if state == self.game.INVALID_MOVE:
            return (self.game.backend.get_state(), -np.infty, False)
        elif state == self.game.GAME_OVER:
            return (self.game.backend.get_state(), self.game.backend.get_last_reward(), True)
    
    def get_state(self):
        return self.game.backend.get_state()
    
    
    def reset(self):
        '''Reset the game'''
        self.game.reset_game()
    