# 2048RL
Deep Reinforcement Learning Model to play 2048


### Running code

To run the code, complete the following steps:
1. Clone the repository using `git clone https://github.com/JohnDLee/2048RL.git`
2. Navitage into the root directory using `cd 2048RL`
3. Create a virtual environemnt, using conda (`conda create -n 2048 python=3.9`) or preferred env manager
4. Run `pip install -e "."` to install the package.
5. Test the backend program using `python3 -m src.backend`

#### Running Scripts
1. Run `python scripts/interactive_game.py` for the interactive game
2. Run `python scripts/random_game.py` for a game using randomly selected moves
3. Run `python scripts/simpleDQN_game.py` for a game using the first iteration of the DQN (poor performance)

### Solution 1 - Deep DQN using an MLP

See here(/RL/SOLUTION1.md) for SOLUTION 1