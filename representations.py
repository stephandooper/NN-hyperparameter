#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GaussianNoise,
    Input,
    MaxPooling2D,
)
from keras.models import Model
import numpy as np


def make_base_repr():
    return {
        'type': None,
        'params': {},
    }

def make_dense_repr():
    layer = make_base_repr()
    layer['type'] = 'dense'
    layer['params']['units'] = 2**np.random.choice(range(3,7))
    return layer


def make_flatten_repr():
    layer = make_base_repr()
    layer['type'] = 'flatten'
    return layer


def make_batchnorm_repr():
    layer = make_base_repr()
    layer['type'] = 'batchnorm'
    return layer


def make_conv2d_repr():
    layer = make_base_repr()
    layer['type'] = 'conv2d'
    layer['params']['filters'] = 2**np.random.choice(range(3, 7))
    layer['params']['kernel_size'] = 3
    layer['params']['activation'] = 'relu'
    return layer


def make_dropout_repr():
    layer = make_base_repr()
    layer['type'] = 'dropout'
    layer['params']['rate'] = np.around(np.random.uniform(low=0.1, high=0.5),
                                        decimals=2)
    return layer


def make_noise_repr():
    layer = make_base_repr()
    layer['type'] = 'noise'
    layer['params']['stddev'] = np.random.random()
    return layer


def make_pool_repr():
    layer = make_base_repr()
    layer['type'] = 'pool'
    layer['params']['pool_size'] = 2
    return layer


def make_conv2d_dropout_repr():
    layer = make_conv2d_repr()
    layer['type'] = 'conv2ddropout'
    layer['params']['rate'] = np.around(np.random.uniform(low=.1, high=.5),
                                        decimals=2)
    return layer


def make_conv2d_pool_repr():
    layer = make_conv2d_repr()
    layer['type'] = 'conv2dpool'
    layer['params']['pool_size'] = 2

    return layer


REPR_MAKERS = {
    'batchnorm': make_batchnorm_repr,
    'conv2d': make_conv2d_repr,
    'conv2ddropout': make_conv2d_dropout_repr,
    'conv2dpool': make_conv2d_pool_repr,
    'dropout': make_dropout_repr,
    'noise': make_noise_repr,
    'pool': make_pool_repr,
    'flatten': make_flatten_repr,
    'dense': make_dense_repr,
}

MUTABLE_PARAMS = {
    'conv2d': [
        'filters'
    ],
    'dropout': [
        'rate'
    ],
}

INSERTABLE = [
    'batchnorm',
    'conv2d',
    'conv2ddropout',
    'dropout',
    'noise',
    'pool',
    'flatten',
    'dense',
]

REPR2LAYER = {
    'batchnorm': BatchNormalization,
    'conv2d': Conv2D,
    'dropout': Dropout,
    'noise': GaussianNoise,
    'pool': MaxPooling2D,
    'flatten': Flatten,
    'dense': Dense,
}


def repr2layer(r):
    return REPR2LAYER[r['type']](**r['params'])


def reprs2nn(reprs):
    inputs = Input(shape=(28, 28, 1))
    x = inputs
    for r in reprs:
        if r['type'] == 'conv2ddropout':
            x = repr2layer({'type': 'conv2d', 'params': {x[0]: x[1] for x in r['params'].items() if x[0] in make_conv2d_repr()['params'].keys()}})(x)
            x = repr2layer({'type': 'dropout', 'params': {x[0]: x[1] for x in r['params'].items() if x[0] in make_dropout_repr()['params'].keys()}})(x)
        elif r['type'] == 'conv2dpool':
            x = repr2layer({'type': 'conv2d', 'params': {x[0]: x[1] for x in r['params'].items() if x[0] in make_conv2d_repr()['params'].keys()}})(x)
            x = repr2layer({'type': 'pool', 'params': {x[0]: x[1] for x in r['params'].items() if x[0] in make_pool_repr()['params'].keys()}})(x)
        else:
            x = repr2layer(r)(x)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)
    return Model(inputs, outputs)


def default_init_nn_repr():
    layers = [
        make_conv2d_repr(),
        make_conv2d_dropout_repr()
    ]
    return layers


def check_validity(reprs):
    start_size = 28
    for layer in reprs:
        t = layer['type']
        p = layer['params']
        if t.startswith('conv2d'):
            start_size -= p['kernel_size'] - 1
            if 'pool' in t:
                start_size /= p['pool_size']
    return start_size >= 1
