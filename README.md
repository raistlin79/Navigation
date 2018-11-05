## Navigation
# DRLND: Deep Reinforcement Learning Project

In this project, an agent is trained to navigate (and collect bananas!) in a large, square world. 

![Environment GIF.](banana.gif)

The simulation contains a single agent that navigates a large environment. At each time step, it has four actions at its disposal:

    0 - walk forward
    1 - walk backward
    2 - turn left
    3 - turn right

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.

A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. However the agent trains until it receives an average of 15 to enhance likelihood of reaching +13 in a trained test run.

 The submission reports the number of episodes needed to solve the environment.

To execute the code you need to have a prebuild unity environment (0.4.0) installed (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip) as well as following packages based on Python 3:
 - Pytorch
 - UnityEnvironment, unityagents
 - Numpy
 - Jupyter

To run and train the agent you can either run DQN_Navigation.ipynb (using jupyter notebook) or python rldqn_navigation.py

The trained network (or better the weights) will be stored in

2018-10-16
