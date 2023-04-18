# 2048RL
Author: John Lee

Deep Reinforcement Learning Model to play 2048


### Running code

To run the code, complete the following steps:
1. Clone the repository using `git clone https://github.com/JohnDLee/2048RL.git`
2. Navitage into the root directory using `cd 2048RL`
3. Create a virtual environemnt, using conda (`conda create -n 2048 python=3.9`) or preferred env manager
4. Run `pip install -e "."` to install the package. (It may be desirable to install the GPU version of Pytorch for actual training.)
5. Test the backend program using `python3 -m src.backend`. It tests a random policy script on the backend.

#### Running Scripts
1. Run `python scripts/interactive_game.py` for the interactive game
2. Run `python scripts/random_game.py` for a game using randomly selected moves
3. Run `python scripts/simpleDQN10000_game.py` for a game using the first iteration of the DQN, trained using the original DQN algorithm for 10k episodes
4. Run `python scripts/everyDQN10000_game.py` for a game using the DQN trained by taking all 4 transitions for a particular state for 10k episodes.
5. Run `python scripts/everyDQN100000_game.py` for the final version of the DQN, trained using the same method as 4. but with 100k episodes.

### Solution 1 - Deep DQN using an MLP

See [here](/RL/SOLUTION1.md) for SOLUTION 1.

### Solution Final - Deep DQN using an MLP observing all rewards for a particular state

See [here](/RL/SOLUTION_FINAL.md) for the final solution.

### Results

Open the notebook at `RL/evaluate_performance.ipynb` to try evaluating each model on 1000 test runs. Otherwise, see Solution Final for final results.