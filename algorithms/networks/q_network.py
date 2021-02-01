"""
Build the fully connected deep q-newtork using tensorflow
Qiangqiang Guo, Jan 8, 2020
"""

import tensorflow as tf
import tensorlayer as tl
import numpy as np

def create_q_network(type, para, name = ''):
    if 's_dim' not in para.keys():
        raise ValueError('Dimision of the state is not defined')
    if 'a_dim' not in para.keys():
        raise ValueError('Dimision of the action is not defined')
    if 'n_hidden_fcn_layers' not in para.keys():
        raise ValueError('Number of hidden fcn layers is not defined')
    if 'n_hidden_fcn_units' not in para.keys():
        raise ValueError('Number of nural nodes of each hidden fcn layer are not defined, please specify it by a list')
    if 'act_funcs_fcn' not in para.keys():
        raise ValueError('Activation functions of each fcn layer are not defined, please specify it by a list')

    s_dim = para['s_dim']
    a_dim = para['a_dim']
    n_hidden_fcn_layers = para['n_hidden_fcn_layers']
    n_hidden_fcn_units = para['n_hidden_fcn_units']
    act_funcs_fcn = para['act_funcs_fcn']
    cnn_para = para['cnn_para']

    if len(n_hidden_fcn_units) != n_hidden_fcn_layers:
        raise ValueError('''Number of hidden layers does not match with the length of hidden node list,
        the length of the latter should be equal to the value of the former''')
    if len(act_funcs_fcn) != n_hidden_fcn_layers + 1:
        raise ValueError('''Number of hidden layers does not match with the length of activation function list,
        the length of the latter should be equal to the value of the former''')

    if type == 'q_fcn': # Fully connected neural network
        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)
        inputs = tl.layers.Input([None, s_dim], name='q_input') # the shape [None, s_dim[0]], first element is the batch size, None means any number will be ok
        x = tl.layers.Dense(n_units = n_hidden_fcn_units[0], act = eval("tf.nn." + act_funcs_fcn[0]), W_init = W_init, b_init = b_init, name = 'hidden_layer_1')(inputs)
        for i in range(1, n_hidden_fcn_layers):
            x = tl.layers.Dense(n_units = n_hidden_fcn_units[i], act = eval("tf.nn." + act_funcs_fcn[i]), W_init = W_init, b_init = b_init, name = '_hidden_layer_' + str(i + 1))(x)
        outputs = tl.layers.Dense(n_units = a_dim, act = eval("tf.nn." + act_funcs_fcn[-1]), W_init = W_init, b_init = b_init, name='A_a')(x)
        return tl.models.Model(inputs=inputs, outputs=outputs, name='q_network_fcn' + name)
    elif type == 'q_cnn':
        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)
        l = inputs = tl.layers.Input((1,) + s_dim)
        for i, kwargs in enumerate(cnn_para):
            l = tl.layers.Conv2d(**kwargs)(l)
        x = tl.layers.Flatten()(l)
        for i in range(1, n_hidden_fcn_layers):
            x = tl.layers.Dense(n_units = n_hidden_fcn_units[i], act = eval("tf.nn." + act_funcs_fcn[i]), W_init = W_init, b_init = b_init, name = '_hidden_layer_' + str(i + 1))(x)
        outputs = tl.layers.Dense(n_units = a_dim, act = eval("tf.nn." + act_funcs_fcn[-1]), W_init = W_init, b_init = b_init, name='A_a')(x)
        return tl.models.Model(inputs=inputs, outputs=outputs, name='q_network_cnn' + name)
