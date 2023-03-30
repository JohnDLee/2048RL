# File: interactive_game.py
# File Created: Friday, 10th February 2023 3:23:20 pm
# Author: John Lee (jlee88@nd.edu)
# Last Modified: Monday, 13th March 2023 12:38:14 am
# Modified By: John Lee (jlee88@nd.edu>)
# 
# Description: Creates a human interactable game of 2048

from src.gui import GAME_2048


if __name__ == '__main__':
    
    game = GAME_2048()
    game.run_interactive_game()