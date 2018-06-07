#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from representation import INSERTABLE, MUTABLE_PARAMS
import numpy as np


def mutate_param(r, param):
    # TODO: implement constraints here
    r[param] = r[param]
    return r


def mutate_param(reprs):
    # Find indices of layers with mutable params
    idxs = []
    for i in range(len(reprs)):
        if reprs[i]['type'] in MUTABLE_PARAMS.keys():
            idxs.append(i)
    # Pick a random index
    idx = np.random.choice(idxs)
    # Pick a random param
    param = np.random.choice(MUTABLE_PARAMS[reprs[idx]['type']])
    r = mutate_param(reprs[idx], param)
    reprs[idx] = r
    return reprs


def insert_layer(reprs):
    # TODO: implement
    pass
