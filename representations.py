#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import (
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
    layer['params']['kernel_initializer'] = 'glorot_normal'
    return layer


def make_flatten_repr():
    layer = make_base_repr()
    layer['type'] = 'flatten'
    return layer


def make_conv2d_repr():
    layer = make_base_repr()
    layer['type'] = 'conv2d'
    layer['params']['filters'] = 2**np.random.choice(range(4, 9))
    layer['params']['kernel_size'] = 3
    layer['params']['kernel_initializer'] = 'glorot_normal'
    layer['params']['activation'] = 'relu'
    return layer


def make_noise_repr():
    layer = make_base_repr()
    layer['type'] = 'noise'
    layer['params']['stddev'] = np.random.random()
    return layer


def make_conv2d_dropout_repr():
    layer = make_conv2d_repr()
    layer['type'] = 'conv2ddropout'
    layer['params']['rate'] = np.around(np.random.uniform(low=.1, high=.5),
                                        decimals=1)
    return layer


def make_conv2d_pool_repr():
    layer = make_conv2d_repr()
    layer['type'] = 'conv2dpool'
    layer['params']['pool_size'] = 2

    return layer


REPR_MAKERS = {
    'conv2d': make_conv2d_repr,
    'conv2ddropout': make_conv2d_dropout_repr,
    'conv2dpool': make_conv2d_pool_repr,
    'noise': make_noise_repr,
    'flatten': make_flatten_repr,
    'dense': make_dense_repr,
}

MUTABLE_PARAMS = {
    'conv2d': [
        'filters'
    ],
    'conv2ddropout': [
        'rate'
    ],
}

INSERTABLE = [
    'conv2d',
    'conv2ddropout',
    'noise',
    'flatten',
    'dense',
]

REPR2LAYER = {
    'conv2d': Conv2D,
    'dense': Dense,
    'dropout': Dropout,
    'flatten': Flatten,
    'noise': GaussianNoise,
    'pool': MaxPooling2D,
}


def repr2layer(r):
    return REPR2LAYER[r['type']](**r['params'])


def reprs2nn(reprs, shape):
    inputs = Input(shape=shape)
    x = inputs
    for r in reprs:
        if r['type'] == 'conv2ddropout':
            x = repr2layer({'type': 'conv2d', 'params':
                           {x[0]: x[1] for x in r['params'].items()
                               if x[0] in make_conv2d_repr()['params'].keys()}})(x)
            x = repr2layer({'type': 'dropout', 'params':
                           {x[0]: x[1] for x in r['params'].items()
                               if x[0] == 'rate'}})(x)
        elif r['type'] == 'conv2dpool':
            x = repr2layer({'type': 'conv2d', 'params':
                           {x[0]: x[1] for x in r['params'].items()
                               if x[0] in make_conv2d_repr()['params'].keys()}})(x)
            x = repr2layer({'type': 'pool', 'params':
                           {x[0]: x[1] for x in r['params'].items()
                               if x[0] == 'pool_size'}})(x)
        else:
            x = repr2layer(r)(x)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)
    return Model(inputs, outputs)


def check_validity(reprs, dataset='fashion'):
    assert dataset in ['fashion', 'cifar10']
    if dataset == 'fashion':
        start_size = 28
    elif dataset == 'cifar10':
        start_size = 32
    for layer in reprs:
        t = layer['type']
        p = layer['params']
        if t.startswith('conv2d'):
            start_size -= p['kernel_size'] - 1
            if 'pool' in t:
                start_size /= p['pool_size']
    return start_size >= 1
