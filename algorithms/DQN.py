"""
implement the DQN agent
Qiangqiang Guo, Jan 9, 2021
"""

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import collections

from networks.q_network import create_q_network

class DQN():
    def __init__(self,
                time_step = 1,
                s_dim = [1,1],
                a_dim = [1,1],
                memory_capacity = 5000,
                epsilon = 0.1,
                batch_size = 64,
                gamma = 0.9,
                n_hidden_layers = 1,
                n_hidden_nodes = [128],
                act_funcs = ['relu', 'softmax'):

        self.time_step = time_step
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.memory_capacity = memory_capacity
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        self.act_funcs = act_funcs

        para = collections.defaultdict()
        para['s_dim'] = self.s_dim
        para['a_dim'] = self.a_dim
        para['n_hidden_layers'] = self.n_hidden_layers
        para['n_hidden_layers'] = self.n_hidden_layers
        para['act_funcs'] = self.act_funcs

        self.q_network = create_q_network('q_fcn', para, 'agent')
        self.q_network.train()

        self.q_learning_target = create_q_network('q_fcn', para, 'target')
        self.copy_para(self.q_learning, self.q_learning_target)
        self.q_learning_target.eval()

    def copy_para(self, from_model, to_model):
        """
        Copy parameters for soft updating
        """
        for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
            j.assign(i)

    def store_transition(self, s, a, r, s_):
        """
        Store transition to memory, delete the earlist one if capacity is exceeded
        """
        
