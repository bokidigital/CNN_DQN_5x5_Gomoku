# CNN_DQN_5x5_Gomoku
CNN deep q network 5x5, 4 to win, gomoku

This is a CNN+DQN reinforcement learning program for 5x5 get 4 Gomoku game.

Please check DEMO_xxx.PNG for the gaming screen where human play ‘O’ and agent play ‘X’ in this game. Agent can win in many conditions as demo captures.

5x5_osc_33cnn_his_20layers.py is training code, user can train his/her own weight from scratch by using this code.

When user want to play with agent, please use _5x5_osc_33cnn_his_20layers_verify.py. Please unzip the trained weights (trained_weigths.zip.001 ~ 005) and put with _5x5_osc_33cnn_his_20layers_verify.py in a same folder.

My test environment:

Tensorflow: 1.14.0

Keras

Python 3.7

Ubuntu 16.04

