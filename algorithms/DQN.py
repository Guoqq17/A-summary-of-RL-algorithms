"""
Implement the DQN agent
Qiangqiang Guo, Jan 9, 2021
"""

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import collections
import random

from algorithms.networks.q_network import create_q_network

class DQN():
    def __init__(self,
                s_dim = 1,
                a_dim = 1,
                memory_capacity = 5000,
                steps_update_target = 200,
                epsilon = 0.1,
                batch_size = 64,
                gamma = 0.9,
                n_hidden_layers = 1,
                n_hidden_units = [128],
                act_funcs = ['relu', 'softmax']):
        '''
        Initialize the DQN agent
        s_dim: dimension of the state space
        a_dim: dimension of the action space
        memory_capacity: the capacity of replay memory
        steps_update_target: for the original DQN, how many steps to wait until update the target network
        epsilon: the greedy action parameter
        batch_size: batch size
        gamma: discount parameter to calculate the expected q value
        n_hidden_layers: number of hidden layers (hidden layers only, e.g., for a 4*128*64*2 network, hidden layers will be 2)
        n_hidden_units: number of unit for each hidden layer, len(n_hidden_units) should be equal to n_hidden_layers
        act_funcs: activation functions for each layers (except the input layer, includes the output layer), len(act_funcs) == n_hidden_layers + 1
        '''
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.memory_capacity = memory_capacity
        self.steps_update_target = steps_update_target
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.act_funcs = act_funcs

        self.pointer = 0
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + 1 + 1), dtype=np.float32)

        para = collections.defaultdict()
        para['s_dim'] = self.s_dim
        para['a_dim'] = self.a_dim
        para['n_hidden_layers'] = self.n_hidden_layers
        para['n_hidden_units'] = self.n_hidden_units
        para['act_funcs'] = self.act_funcs

        self.q_network = create_q_network('q_fcn', para, 'agent')
        self.q_network.train()

        self.q_network_target = create_q_network('q_fcn', para, 'target')
        self.copy_para(self.q_network, self.q_network_target)
        self.q_network_target.eval()

    def copy_para(self, from_model, to_model):
        """
        Update target q network parameters
        """
        for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
            j.assign(i)

    def store_transition(self, s, a, r, s_):
        """
        Store transition to memory
        """
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def greedy_action(self, s):
        """
        Choose action based on current state (s) using the epsilon-greedy strategy
        """
        action_candidate = self.q_network(np.array([s],dtype=np.float32)).numpy().tolist()
        if random.random() < self.epsilon:
            return random.randint(0, self.a_dim - 1)
        else:
            return np.argmax(action_candidate)
        return action_candidate

    def learn(self):
        """
        Train the q network by backproporgation
        """
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + 1]
        br = bt[:, self.s_dim + 1]
        bs_ = bt[:, -self.s_dim:]

        q_value = self.q_network(bs).numpy()
        next_target_q_value = tf.reduce_max(self.q_network_target(bs_), axis = 1)
        y = br + self.gamma * next_target_q_value
        q_value[range(self.batch_size), np.array(ba, dtype = int)] = y

        with tf.GradientTape() as tape:
            q_value_pred = self.q_network(bs)
            loss = tf.losses.mean_squared_error(q_value, q_value_pred)
        grads = tape.gradient(loss, self.q_network.trainable_weights)
        self.q_network_opt = tf.optimizers.Adam()
        self.q_network_opt.apply_gradients(zip(grads, self.q_network.trainable_weights))

        if not (self.pointer - self.memory_capacity) % self.steps_update_target:
            self.copy_para(self.q_network, self.q_network_target)
