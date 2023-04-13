# File: backend.py
# File Created: Friday, 10th February 2023 2:48:38 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Thursday, 13th April 2023 5:46:30 pm
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Backend for the 2048 game for quick interactions in RL

import numpy as np

class BACKEND_2048():
    """Backend for 2048 game
    """
    SUCCESSFUL_MOVE=0
    INVALID_MOVE=1
    GAME_OVER=2
    
    
    
    def __init__(self) -> None:
        self.init_game()
        
    def get_state(self):
        """Gets state
        """
        return self.state
    
    def get_last_reward(self):
        """Gets reward from last movement.
        """
        return self.last_reward

    def get_score(self):
        """Gets score"""
        return self.score
    
    def get_last_compressed(self):
        return self.last_compressed
        
    def init_game(self):
        """Initialize state w/ random 2's
        """
        self.state = np.zeros((4, 4), dtype=int)
        self.num_vals = 0
        self.score = 0
        self.num_operations = 0
        self.last_reward = 0
        self.spawn_tile()
        self.spawn_tile()
    
    def spawn_tile(self):
        """Spawns a tile in an empty location
        """
        # new tile of 2 or 4
        new_tile_val = np.random.randint(1, 3) * 2 
        # find an empty location (val = 0) and set it to the new tile val
        x, y = np.where(self.state == 0)
        idx = np.random.choice(x.shape[0])
        self.state[x[idx], y[idx]] = new_tile_val
        self.num_vals += 1
        
    ############
    # Left Ops #
    ############
    
    def stack_left(self):
        """Shifts all terms to the left"""
        stacked = 0
        for idx, arr in enumerate(self.state):
            nonzero = arr.nonzero()[0]
            tmp = np.arange(len(nonzero))
            if len(nonzero) > 0 and not np.array_equal(nonzero, tmp):
                self.state[idx][np.arange(len(nonzero))] = arr[nonzero]
                self.state[idx][np.arange(len(nonzero), 4)] = 0
                stacked = True
        # if stack was moved
        return stacked
    
    def compress_left(self):
        """Compresses duplicates"""
        # compress items.
        compressed = False
        self.last_reward = 0
        self.last_compressed = 0
        for i in range(3):
            idxs = np.where(self.state[:,i]==self.state[:,i+1])[0]
            self.last_compressed += len(idxs)
            non_zeros = np.where(self.state[idxs, i] != 0)[0]
            if len(non_zeros) == 0:
                continue
            self.state[idxs, i] *= 2
            self.state[idxs, i+1] = 0
            self.last_reward += self.state[idxs, i].sum()
            self.num_vals -= len(non_zeros)
            compressed = True
        self.score += self.last_reward
        # if somewhere was compressed, return True
        return compressed

    def left(self,):
        """Performs left operation in game"""
        stacked = self.stack_left()
        if not self.compress_left() and not stacked: return self.INVALID_MOVE
        self.stack_left()
        self.num_operations += 1
        self.spawn_tile()
        if self.check_game_over(): return self.GAME_OVER
        return self.SUCCESSFUL_MOVE
        
    #############
    # Other ops #
    #############
    def reverse(self):
        """Reverses matrix"""
        self.state = np.fliplr(self.state)
    
    def transpose(self):
        """Transposes Matrix"""
        self.state = self.state.T
    
    def right(self):
        """Performs Right operation"""
        # reverse to get left-equivalent op
        self.reverse()
        stacked = self.stack_left()
        if not self.compress_left() and not stacked: 
            self.reverse()
            return self.INVALID_MOVE
        self.num_operations += 1
        self.stack_left()
        self.reverse()
        self.spawn_tile()
        if self.check_game_over(): return self.GAME_OVER
        return self.SUCCESSFUL_MOVE
        
    def up(self):
        """Performs Up operation"""
        # transpose to get left-equivalent op
        self.transpose()
        stacked = self.stack_left()
        if not self.compress_left() and not stacked:
            self.transpose()
            return self.INVALID_MOVE
        self.num_operations += 1
        self.stack_left()
        self.transpose()
        self.spawn_tile()
        if self.check_game_over(): return self.GAME_OVER
        return self.SUCCESSFUL_MOVE
        
    def down(self):
        """Perform down operaion"""
        # Transpose & Reverse to get left-equivalent op
        self.transpose()
        self.reverse()
        stacked = self.stack_left()
        if not self.compress_left() and not stacked:
            self.reverse()
            self.transpose()
            return self.INVALID_MOVE
        self.num_operations += 1
        self.stack_left()
        self.reverse()
        self.transpose()
        self.spawn_tile()
        if self.check_game_over(): return self.GAME_OVER
        return self.SUCCESSFUL_MOVE
    
    ###################
    # Check Game Over #
    ###################
    
    def check_game_over(self):
        """ Check if there is no valid move left"""
        if self.num_vals == 16: # only if board is full.
            for i in range(4):
                for j in range(3):
                    if self.state[i][j] == self.state[i][j+1]:
                        return False
                    if self.state[j][i] == self.state[j+1][i]:
                        return False
            return True
        return False
                    
    
if __name__ == '__main__':
    # test untill failure
    test = BACKEND_2048()
    while True:
        print(test.state)
        if test.right() == test.GAME_OVER: break
        print(test.state)
        if test.up() == test.GAME_OVER: break
        print(test.state)
        if test.left() == test.GAME_OVER: break
        print(test.state)
        if test.down() == test.GAME_OVER: break
    print("Game over, your score was:", test.get_score())
    
    
    