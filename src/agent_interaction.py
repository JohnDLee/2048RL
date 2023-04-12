
from src.gui import GAME_2048
from src.backend import BACKEND_2048
import numpy as np

class ENV2048():
    
    def __init__(self, gui = True):
        # start a game
        self.gui = gui
        if self.gui:
            self.game = GAME_2048()
            self.backend = self.game.backend
        else:
            self.game = BACKEND_2048()
            self.backend = self.game
        
        self.actions = [self.game.left, self.game.up, self.game.right, self.game.down]
        
        
    def num_actions(self):
        ''' gives the total number of actions possible '''
        return len(self.actions)
    
    def random_action(self):
        ''' gives a random action to do '''
        return np.random.randint(4) # 0 - 3
    
    def pseudo_step(self, action):
        """Takes a pseudo step
        """
        # save previous state
        cur_state = self.backend.get_state()
        cur_score = self.backend.get_score()
        num_vals = self.backend.num_vals
        
        results = self.step(action)
        # reload it back
        self.backend.state = cur_state
        self.backend.score = cur_score
        self.backend.num_vals = num_vals
        self.backend.num_operations -= 1
        return results
        
        
    
    def step(self, action):
        ''' takes an action,
        returns (next_state, reward, game_over)'''
        state = self.actions[action](None) # get state ccorresponding to action
        
        if state == self.game.SUCCESSFUL_MOVE:
            return (self.backend.get_state(), self.backend.get_last_reward(), False, state)
        if state == self.game.INVALID_MOVE:
            return (self.backend.get_state(), -512, False, state)
        elif state == self.game.GAME_OVER:
            return (self.backend.get_state(), self.backend.get_last_reward(), True, state)
    
    def get_state(self):
        return self.backend.get_state()
    
    
    def reset(self):
        '''Reset the game'''
        if self.gui:
            self.game.reset_game()
            # need to reassign backend
            self.backend = self.game.backend
        else:
            del self.game
            self.game = BACKEND_2048()
            self.backend = self.game
            self.actions = [self.game.left, self.game.up, self.game.right, self.game.down]
            
        return self.backend.get_state()
    
  