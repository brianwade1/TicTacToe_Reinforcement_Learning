# Tic-Tac-Toe reinforcement learning practice

This repo teaches an agent to play tic-tac-toe using the standard q-learning algorithm. Users can play against another human or the trained agent. When a human plays against the agent, the agent uses imitation learning and uses the game history and outcome to update its policy.

![tictactoe](/images/tictactoe_screenshot.png)

## Folders and Files

This repo contains the following folders and files:

* [agents](agents) : Trained agents
   * Q_learning_TicTacToe_policy: Trained agent policy (state-value pairs) stored as a pickle object.

* [images](images) : Images used in this Readme file
   * [tictactoe_screenshot.png](tictactoe_screenshot.png): Screen shot from testing the agent.

* [game.py](game.py) - Main script for the tic-tac-toe game class and all of its methods. Running this file as main allows the user to play against another human.

* [play_agent.py](play_agent.py) - This script allows the user to play the agent. When run as main, the user can select X or O and then the game begins with a random toss of who goes first: the human or agent. 

* [Q_learning_agent.py](Q_learning_agent.py) - Main script for the Q-Learning-Agent class. When run as main, the agent plays 500,000 games against itself while updating its policy using the standard q-learning algorithm.

* [settings.py](settings.py) - Script that includes all the user settings for the tic-tac-toe game, q-learning agent, and training process.

---
## References

1. Sutton, Richard S. and Barto, Andrew G. <ins>Reinforcement Learning: An Introduction</ins>, 2nd ed. The MIT Press; Cambridge, Massachusetts. 2018