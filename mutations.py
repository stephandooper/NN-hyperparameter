#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from representations import (INSERTABLE,
                             MUTABLE_PARAMS,
                             REPR_MAKERS,
                             check_validity)
import representations
import numpy as np
import random

containerIndividual = None
getRandomLayer = None


def setIndividual(container):
    '''
    Sets the individual Object to the mutations global class.
    In order to generate a representation of type 'Individual', you need to call the variable.
    Example:
        containerIndividual(array) --> Individual([block1], [block2], [block3])
    '''
    global containerIndividual
    containerIndividual = container


def setInitialization(func):
    global getRandomLayer
    getRandomLayer = func


def mutate_layer(layer, verbose=False):
    '''
    Looks up the initializer function for a type,
    and replaces it with a new initialization.

    Appended [0] because you need to access the container inside the
    Individual({type: conv2dpool, params})
    class.
    '''
    for elem in REPR_MAKERS:
        if layer['type'] == elem:
            layer = REPR_MAKERS[elem]()
            if verbose:
                print('mutate layer {}'.format(layer['type']))
    return layer


def mutate_network(reprRaw, mutations=1, verbose=False, appendRemoveProb=0.6):
    '''
    Mutates a whole representation network.
    Arguments:
        appendRemoveProb: Probability that a block is being appended/removed.
            The inverse is the probability of mutating a block itself (so not
            changing the structure)
    '''
    repr = reprRaw.tolist()

    if mutations > len(repr):
        mutations = len(repr)

    if verbose:
        print("MUTATING %d BLOCKS OF NETWORK" % mutations)

    if np.random.random() < appendRemoveProb:
        if np.random.random() < 0.5 or len(repr) <= 2:
            repr.append(getRandomLayer())
            if verbose:
                print("APPEND BLOCK ELEM: ", len(repr))
                print("NEW BLOCK ELEM: ", len(repr))
        else:
            random.shuffle(repr)
            repr.pop()
            if verbose:
                print("REMOVE BLOCK ELEM: ", len(repr))
                print("NEW BLOCK ELEM: ", len(repr))
    else:
        for layerIndex in np.random.randint(0, len(repr), mutations):
            repr[layerIndex] = mutate_layer(repr[layerIndex], verbose=verbose)

    while not check_validity(repr):
        random.shuffle(repr)
        repr.pop()

    if verbose:
        print('Network valid: {}'.format(check_validity(repr)))

    return containerIndividual(repr),
