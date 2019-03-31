# Multy Agent Deep Deterministic Policy Gradient Tennis Unity Env Report

## Architecture

### Multy-Agent

### Deep Determinist Policy Gradient (DDPG)

The environment in which this model has been trained is based on one single double-joined arm. The code can be adapted easily to 20 jointed-arms architecture. The DDPG is a model which iterativelly learns an optimal policy using the Q-Learning algorithm. Q-Learning is reached thanks to the usage of Bellman equation:

![Bellman](https://spinningup.openai.com/en/latest/_images/math/339d9f6adec072789c579d36f9d1791e6246b075.svg)

DDPG learns the optimal action-value function and then define the best action to perform. When you have an optimal action-value in a discrete space is particular easy to identify the best action. In a continous space the things change a little bit. In this latter, you may need a differentiable action-space function in any point in order to adopt a gradient approach for the identification of the right action. 

The main goal of DDPG based on a neural network is the minimization of the mean-squared Bellman error (MSBE). This function describe how closely the action-value function is going to satisfy the BellMan Equation

![Error](https://spinningup.openai.com/en/latest/_images/math/d193a1fae2f39357adc458987f0301518f3cd669.svg)

The DDPG classical implementation uses several methodologies / tricks for improving their performances:

* **Replay Buffer:** The models exploit a buffer of previous experiences in order to learn from uncorrelated events.

* **Two Target Agent:** Due to the influenced weight updates a common approach is to fix them in another network in order to maintain a fixed target. The main reason is associate to the fact that we are training over the same parameters which are used for MSBE minimization.

* **Ornstein–Uhlenbeck Noise:** This is an approach used for defining the exploration and exploitation trade-off. 

### PyTorch Network

The PyTorch architecture is based on a Feed-Forward Neural Network with two hidden layer for both actor and critic. The first hidden layer is 256 and the second is 128 hidden units. 

The Actor Model:
* ```Hidden: (input*2, 256) - ReLU```
* ```Hidden: (256, 128) - ReLU```
* ```Output: (128, 4) - TanH```

The Critic Model:
* ```Hidden: (input*2, 256) - ReLU```
* ```Hidden: (256 + action_size*2, 128) - ReLU```
* ```Output: (128, 1) - Linear```

![FFNN](http://neuralnetworksanddeeplearning.com/images/tikz11.png)

## HyperParameters

This is the list of hyperparameters that has been used for the model training procedure. These values are also the inside the ddpg_agent.py file code global variables. 

*  ```BUFFER_SIZE = int(1e6)  # replay buffer size```
* ```BATCH_SIZE = 256       # minibatch size```
* ```GAMMA = 0.99            # discount factor```
* ```TAU = 1e-3              # for soft update of target parameters```
* ```LR_ACTOR = 2e-4         # learning rate of the actor ```
* ```LR_CRITIC = 3e-4        # learning rate of the critic```
* ```WEIGHT_DECAY = 0        # L2 weight decay```
* ```ITERATION_LEARNING = 2  # number of times to iterate the training process```
* ```FREQUENCY_LEARNING = 1  # after how many steps it is required a training process```
* ```NOISE_OU = 0.25         # noise mean ```
* ```NOISE_SIGMA = 0.2        # Sigma value of OUNoise ```


### Results
