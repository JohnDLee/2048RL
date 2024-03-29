# Solution 1 - Deep Q-Network

## Environment 

The 2048 environment is set up as follows.
In 2048, a 4x4 board of numbers in multiples of 2 and there are 4 possible actions to take, (Left Up, Right, Down). The 4x4 board consists of the entire game state at any point in time. At initialization, the board contains 2 2/4's at any position.

There are 2 kind of moves - Successful and unsuccessful. If a successful move is made, numbers are shifted to their corresponding directions and stacked togethor into a higher multiple of 2. Then, a new 2/4 is randomly placed into an open location.
If an unsuccessful move was made, it meands no numbers could shift and be stacked, and the game state remains the same.

After making each action, there is an associated reward corresponding to the value of the numbers that get stacked. For example, stacking 2 128 blocks gives a score of 256, whereas stacking 2 4 blocks only gives 8. Furthermore, an unsuccessful move is implemented with a reward of -512, attempting to discourage the agent from making them.

The game is played until no successful move can be made.

## Training Method

In my first solution, I took a Deep Q-Learning approach to solving 2048. The implementation of DQN can be found [here](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html). Essentially, I try using the policy model to determine the reward of the next possible move, selecting the move that gives the largest possible reward. 


The first iteration model was only trained on 500 episodes, but I anticipate this to be an insufficient number of episodes. Future models will be trained longer.

Training can be run using `python RL/training_simple [-gui] [-eps N]` to train with GUI on or off and plotting.

## Model

The model I use is a simple MLP with 16 input neurons, 1024, 512, 256 hidden neurons, and 4 output neurons. The internal activations are ReLU operations and the output is just left linear. States are flatten to a 16 value vector to be fed into the model and each output neuron corresponds to the reward of a certain move. After doing research on other similar projects, this model is of sufficient size to capture the 2048 problem.

An alternative model that could be used is a convolutional network, since the 4x4 board could be treated like a "grayscale" image.

For an optimization function, I use an Adam algorithm, which was shown to be efficient at converging. However, I am not entirely sure the optimization algorithm matters significantly in this case, so picking Adam is just a heuristic.

For the loss function, I use the Huber Loss, because it take the best traits of MSE and MAE. This way significant losses are punished greater, but not too greatly. I could also just use the plain Huber Loss.

## Results

### Simple DQN
The training results over episodes is shown as follows. The main performance metric is the final score of the episode. While the score creeps up initially, around 800, we see that it drops and doesn't proceed to learn. 

The loss profile is as follows:

![train_loss_profile](../trained_models/simpleDQN10000/train_results.png)



I ran 1000 test simulations of the game and the results are below.
![test_results](../results/RMT_simple1000.png)


| Metric | Mean | Std | Min | Max |
| ------ | ---- | --- | --- | --- |
| Score  | 558.96 | 362.11 | 36 | 2568 | 
| Moves  | 62.91 | 23.78 | 19 | 163 | 
| Top Squares | - | - | 8 | 256 | 

| Value | Counts |
| ----- | ------ | 
| 8 | 1 | 
| 16 | 82 | 
| 32 | 358 | 
| 64 | 417 | 
| 128 | 134 | 
| 256 | 8 |

### Random

We can actually compare it to a random strategy and see that it actually does worse.

Using a 1000 simulations of the game, the statistics are as follows:

![random_results](../results/RMT_random.png)

| Metric | Mean | Std | Min | Max |
| ------ | ---- | --- | --- | --- |
| Score  | 909.85 | 362.11 | 40 | 2884 | 
| Moves  | 84.03 | 27.77 | 20 | 193 | 
| Top Squares | - | - | 8 | 256 | 

| Value | Counts |
| ----- | ------ | 
| 8 | 1 | 
| 16 | 3 | 
| 32 | 97 | 
| 64 | 434 | 
| 128 | 407 | 
| 256 | 58 |


## Running using the Trained Model

A Monte-Carlo simulation of 1000 test runs can be observed in the notebook at `RL/evaluate_performance.ipynb`.

The test run using the trained model can be accomplished by running `python scripts/simpleDQN10000_game.py`.
A test run using a random policy can be accomplished by running `python scripts/random_game.py`.

## Improvement

There are many areas that could be considered for improvement.
First, we might check wheter a convolutional neural network works better. It may or may not, since the weights being multiplied are essentially the same, but it might add a bit of positional relevance.

A more drastic option is to change away from a DQN training method, perhaps using double DQN or Dueling DQNs. Another option is to evaluate the reward of every possible move from a certain state, even perhaps using a Monte Carlo simulation to model the probabilistic nature of new 2/4 blocks.

Lastly, we will train the model for a much longer time and observe whether the results improve even more.



