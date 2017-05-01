# Flappy Bird - Deep Reinforcement Learning

Introduction to Deep Reinforcement Learning with the game Flappy Bird.

## Installation dependencies
* Python 2.7
* TensorFlow >= 1.0
* pygame
* OpenCV-Python

## How to train ?
```
git clone https://github.com/xebia-france/intro-reinforcement-learning.git
cd intro-reinforcement-learning/deep_rf_flappy_bird
python deep_q_network.py 
```
This will launch the training with default parameters

## How to play ?
Once you have trained your network, you can launch the demo using the checkpoints saved during training.
```
python play_game.py --checkpoint_directory="path/to/checkpoint_directory" 
```

## Disclaimer
This work is highly based on the following repo : https://github.com/yenchenlin/DeepLearningFlappyBird
