"""
Build the fully connected deep q-newtork using tensorflow
Qiangqiang Guo, Jan 8, 2020
"""

import tensorflow as tf
import tensorlayer as tl
import numpy as np

def create_q_network(type, para, name = ''):
    if type == 'q_fcn': # Fully connected neural network
        if 's_dim' not in para.keys():
            raise ValueError('Dimision of the state is not defined')
        if 'a_dim' not in para.keys():
            raise ValueError('Dimision of the action is not defined')
        if 'n_hidden_layers' not in para.keys():
            raise ValueError('Number of hidden layers is not defined')
        if 'n_hidden_units' not in para.keys():
            raise ValueError('Number of nural nodes of each hidden layer are not defined, please specify it by a list')
        if 'act_funcs' not in para.keys():
            raise ValueError('Activation functions of each layer are not defined, please specify it by a list')

        s_dim = para['s_dim']
        a_dim = para['a_dim']
        n_hidden_layers = para['n_hidden_layers']
        n_hidden_units = para['n_hidden_units']
        act_funcs = para['act_funcs']

        if len(n_hidden_units) != n_hidden_layers:
            raise ValueError('''Number of hidden layers does not match with the length of hidden node list,
            the length of the latter should be equal to the value of the former''')
        if len(act_funcs) != n_hidden_layers + 1:
            raise ValueError('''Number of hidden layers does not match with the length of activation function list,
            the length of the latter should be equal to the value of the former''')

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)
        inputs = tl.layers.Input([None, s_dim], name='q_input') # the shape [None, s_dim[0]], first element is the batch size, None means any number will be ok
        x = tl.layers.Dense(n_units = n_hidden_units[0], act = eval("tf.nn." + act_funcs[0]), W_init = W_init, b_init = b_init, name = 'hidden_layer_1')(inputs)
        for i in range(1, n_hidden_layers):
            x = tl.layers.Dense(n_units = n_hidden_units[i], act = eval("tf.nn." + act_funcs[i]), W_init = W_init, b_init = b_init, name = '_hidden_layer_' + str(i + 1))(x)
        outputs = tl.layers.Dense(n_units = a_dim, act = eval("tf.nn." + act_funcs[-1]), W_init = W_init, b_init = b_init, name='A_a')(x)
        return tl.models.Model(inputs=inputs, outputs=outputs, name='q_network_fcn' + name)
