#!/usr/bin/env python3
# -*- coding: utf-8 -*-



'''
MAKE SURE TO RETURN TYPE 'Individual' (toolbox.individual())

It is safe to mutate droprates/pooling stuff, etc. But inserting/removing seems impossible?
(Type 'Individual' has no function 'remove'/'append')
'''




from representations import INSERTABLE, MUTABLE_PARAMS, default_init_nn_repr, REPR_MAKERS
import representations
import numpy as np
import random

FILTER_MUTATIONS = np.array([16, 32, 64])

containerIndividual = None
getRandomLayer = None



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

# -------------------------------------------------------



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
                print('MUTATED LAYER %s' % layer['type'])
    return layer


def removeBlock(repr):
    #@Deprecated
    random.shuffle(repr)
    return repr.pop()


def mutate_network(reprRaw, mutations=2, verbose=False, appendProb = 0.5, removeProb = 0.5):
    '''
    Mutates a whole representation network.
    Arguments:
        appendProb: probability that a block is being appended/removed.
    '''
    repr = reprRaw.tolist()

    if mutations > len(repr):
        mutations = len(repr) # prevents setting higher count of mutations than length of representation


    if verbose:
        print("MUTATING %d BLOCKS OF NETWORK" % mutations)
    
    
    #Mutating blocks itself:
    for layerIndex in np.random.randint(0, len(repr), mutations):
        if verbose:
            repr[layerIndex] = mutate_layer(repr[layerIndex], verbose=True)
        else:
            repr[layerIndex] = mutate_layer(repr[layerIndex])

    #Appending / removing blocks:
    if np.random.random() < appendProb:
        print("APPEND BLOCK ELEM: ", len(repr))
        repr.append(getRandomLayer())
        print("NEW BLOCK ELEM: ", len(repr))
    elif np.random.random() < removeProb and len(repr) > 2:
        print("REMOVE BLOCK ELEM: ", len(repr))
        random.shuffle(repr)
        repr.pop()
        print("NEW BLOCK ELEM: ", len(repr))


    return containerIndividual(repr),