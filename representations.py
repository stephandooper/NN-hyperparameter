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


def make_batchnorm_repr():
    layer = make_base_repr()
    layer['type'] = 'batchnorm'
    return layer


def make_conv2d_repr():
    layer = make_base_repr()
    layer['type'] = 'conv2d'
    layer['params']['filters'] = 2**5
    layer['params']['kernel_size'] = 3
    layer['params']['activation'] = 'relu'
    return layer


def make_dropout_repr():
    layer = make_base_repr()
    layer['type'] = 'dropout'
    layer['params']['rate'] = np.around(np.random.uniform(low=0.1, high=0.5),
                                        decimals=1)
    return layer


def make_noise_repr():
    layer = make_base_repr()
    layer['type'] = 'noise'
    layer['params']['stddev'] = .5
    return layer


def make_pool_repr():
    layer = make_base_repr()
    layer['type'] = 'pool'
    layer['params']['pool_size'] = 2
    return layer


def make_input_repr():
    layer = make_base_repr()
    layer['type'] = 'input'
    layer['params']['shape'] = (28, 28, 1)
    return layer


def make_output_repr():
    layer = make_base_repr()
    layer['type'] = 'output'
    layer['params']['units'] = 10
    layer['params']['activation'] = 'softmax'
    return layer


def make_flatten_repr():
    layer = make_base_repr()
    layer['type'] = 'flatten'
    return layer


REPR_MAKERS = {
    'batchnorm': make_batchnorm_repr,
    'conv2d': make_conv2d_repr,
    'dropout': make_dropout_repr,
    'flatten': make_flatten_repr,
    'input': make_input_repr,
    'noise': make_noise_repr,
    'output': make_output_repr,
    'pool': make_pool_repr,
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
    'dropout',
    'noise',
    'pool',
]

REPR2LAYER = {
    'batchnorm': BatchNormalization,
    'conv2d': Conv2D,
    'dropout': Dropout,
    'flatten': Flatten,
    'input': Input,
    'noise': GaussianNoise,
    'output': Dense,
    'pool': MaxPooling2D,
}


def repr2layer(r):
    return REPR2LAYER[r['type']](**r['params'])


def reprs2nn(reprs):
    assert reprs[0]['type'] == 'input' and reprs[-1]['type'] == 'output'

    inputs = repr2layer(reprs[0])
    x = inputs
    for r in reprs[1:]:
        x = repr2layer(r)(x)
    return Model(inputs, x)


def default_init_nn_repr():
    layers = [
        make_input_repr(),
        make_conv2d_repr(),
        make_conv2d_repr(),
        make_flatten_repr(),
        make_output_repr(),
    ]
    return layers
