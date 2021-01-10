"""
implement the DQN agent
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
                time_step = 1,
                s_dim = 1,
                a_dim = 1,
                memory_capacity = 5000,
                steps_update_target = 200,
                epsilon = 0.1,
                batch_size = 64,
                gamma = 0.9,
                n_hidden_layers = 1,
                n_hidden_nodes = [128],
                act_funcs = ['relu', 'softmax']):

        self.time_step = time_step
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.memory_capacity = memory_capacity
        self.steps_update_target = steps_update_target
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        self.act_funcs = act_funcs

        self.pointer = 0
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + 1 + 1), dtype=np.float32)

        para = collections.defaultdict()
        para['s_dim'] = self.s_dim
        para['a_dim'] = self.a_dim
        para['n_hidden_layers'] = self.n_hidden_layers
        para['n_hidden_nodes'] = self.n_hidden_nodes
        para['act_funcs'] = self.act_funcs

        self.q_network = create_q_network('q_fcn', para, 'agent')
        self.q_network.train()

        self.q_network_target = create_q_network('q_fcn', para, 'target')
        self.copy_para(self.q_network, self.q_network_target)
        self.q_network_target.eval()

    def copy_para(self, from_model, to_model):
        """
        Copy parameters for soft updating
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
        action_candidate = self.q_network(np.array([s],dtype=np.float32)).numpy().tolist()
        if random.random() < self.epsilon:
            return random.randint(0, self.a_dim - 1)
        else:
            return np.argmax(action_candidate)
        return action_candidate

    def learn(self):
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + 1]
        br = bt[:, self.s_dim + 1]
        bs_ = bt[:, -self.s_dim:]

        q_value = self.q_network(bs).numpy()
        next_target_q_value = tf.reduce_max(self.q_network_target(bs_), axis = 1)
        y = br + self.gamma * next_target_q_value
        # print(int(ba))
        q_value[range(self.batch_size), np.array(ba, dtype = int)] = y

        with tf.GradientTape() as tape:
            q_value_pred = self.q_network(bs)
            loss = tf.losses.mean_squared_error(q_value, q_value_pred)
        grads = tape.gradient(loss, self.q_network.trainable_weights)
        self.q_network_opt = tf.optimizers.Adam()
        self.q_network_opt.apply_gradients(zip(grads, self.q_network.trainable_weights))

        if not (self.pointer - self.memory_capacity) % self.steps_update_target:
            self.copy_para(self.q_network, self.q_network_target)
