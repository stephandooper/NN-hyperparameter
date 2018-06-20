#!/usr/bin/env python3
# -*- coding: utf-8 -*-



'''
MAKE SURE TO RETURN TYPE 'Individual' (toolbox.individual())

It is safe to mutate droprates/pooling stuff, etc. But inserting/removing seems impossible?
(Type 'Individual' has no function 'remove'/'append')
'''




from representations import INSERTABLE, MUTABLE_PARAMS, default_init_nn_repr, REPR_MAKERS
import numpy as np
import random

FILTER_MUTATIONS = np.array([16, 32, 64])


def mutate_random_param(reprs):
    #@Deprecated
    '''
    main function to the mutate random parameters for a representation  reprs
    
    '''
    # Find indices of layers with mutable params
    idxs = []
    for i in range(len(reprs)):
        if reprs[i]['type'] in MUTABLE_PARAMS.keys():
            idxs.append(i)
    # Pick a random index
    idx = np.random.choice(idxs)
    # Pick a random param
    param = np.random.choice(MUTABLE_PARAMS[reprs[idx]['type']])
    print('param is equal to', param)
    print('repr before is equal to',reprs[idx]['params'][param])
    r = mutate_filter(reprs[idx], param)
    
    reprs[idx] = r
    print('repr after is equal to',reprs[idx]['params'][param])
    return reprs


def insert_layer(reprs):
    #@Deprecated
    # TODO: implement
    return reprs

def mutate_append_remove(reprs, prob_remove=1):
    #@Deprecated
    '''
    Randomly appends or removes a layer. 
        prob_remove: probability that it removes a layer
    Returns: mutated representation
    '''
    if np.random.random() < prob_remove:
        if len(reprs) > 0:
            reprs.remove(np.random.choice(reprs))
    else:
        reprs = insert_layer(reprs)
    return reprs

def mutate_layer(layer, verbose=False):
    '''
    Looks up the initializer function for a type,
    and replaces it with a new initialization.

    Appended [0] because you need to access the container inside the
    Individual({type: conv2dpool, params})
    class.    
    '''
    for elem in REPR_MAKERS:
        if layer[0]['type'] == elem:
            layer[0] = REPR_MAKERS[elem]()
            if verbose:
                print('MUTATED LAYER %s' % layer[0]['type'])
    return layer,

def mutate_network(repr, mutations=1, verbose=False):
    '''
    @Deprecated
    Mutates a whole representation network.
    Apparently, the mutate function of deap does select a layer of itself to mutate, not a whole network.
    '''
    if mutations > len(repr):
        mutations = len(repr)-1 # prevents setting higher count of mutations than length of representation

    if verbose:
        print("MUTATING %d BLOCKS OF NETWORK" % mutations)
    
    for layerIndex in np.random.randint(0, len(repr), mutations):
        if verbose:
            repr[layerIndex] = mutate_layer(repr[layerIndex], verbose=True)
        else:
            repr[layerIndex] = mutate_layer(repr[layerIndex])
    return repr,