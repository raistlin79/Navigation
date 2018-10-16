## Navigation Project Report
DRLND: Deep Reinforcement Learning Project

The report at hand shall provide an overview of the applied algorithms as well as opportunities to further evolve the agent and the neural network behind.

# Learning Algorithm

The Agent contains two fully connected Q-Networks, a local and a target network, implementing a Double Network. The target Q-value of the local network is based on the received reward as well as the discounted Q-value of the next state. This Q-value is derived from the parameters of the target network and the action chosen by the local network.

Local and target network are realising the Duelling Network approach. Therefore the classical hidden layers a followed by two additional streams which results are combined to the output layer.

The ReplayBuffer stores experiences (of size BUFFER_SIZE) and allows the agent to randomly sample from these experiences. Experiences are treated uniformly.

The following hyper parameters are used:
BUFFER_SIZE = int(1e5)   # replay buffer size
BATCH_SIZE = 64          # minibatch size
GAMMA = 0.99             # discount factor
TAU = 1e-3               # for soft update of target parameters
LR = 5e-4                # learning rate
UPDATE_EVERY = 4         # how often to update the network
SEED = 0                 # Providing seed for Qnetworks and ReplayBuffer

The network structure is as follows:
qnetwork_local
                                                                              -> 64 (val - duelling stream)
37 (input layer = states_sizes) -> 64 (hidden layer) -> 128 (hidden layer)                                      -> 4 (actions_size - output layer)
                                                                              -> 64 (adv - duelling stream)

qnetwork_target
                                                                              -> 64 (val - duelling stream)
37 (input layer = states_sizes) -> 64 (hidden layer) -> 128 (hidden layer)                                      -> 4 (actions_size - output layer)
                                                                              -> 64 (adv - duelling stream)

More explanation is provided in the code itself.

# Plot of Rewards
A plot of reward is calculated every time you train the agent using DQN_Navigation.ipynb.

The plot for reaching +13 (in 523 episodes) is given in
 - 181016_plot_of_rewards_13_523.png
Weights are provided in trained_neuralnetwork_13.pth

A run with the submitted implementation aiming at +15 is provided under
 - 181016_plot_of_rewards_15_635.png
The agent reached an average score of +15 after 635 episodes.
Weights are provided in trained_neuralnetwork_13.pth

# Ideas for Future Work
Only two optimization functions have been realized so far. So
 - Prioritized experience replay
 - multi-step bootstrap targets
 - Distributional DQN
 - Noisy DQN
 can still be added to further enhance the agent.

 But even with the existing model hyper parameters can still be adjusted to improve Neural Network performance, specially layer sizes and
  - discount factor (GAMMA)
  - learning rate (LR)
  - update cycle (UPDATE_EVERY)  
  - adapting epsilon function
  - change seed value
  - change activation function

2018-10-16
