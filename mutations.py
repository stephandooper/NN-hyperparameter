#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Delete keras, default_init_nn_repr


from representations import INSERTABLE, MUTABLE_PARAMS, default_init_nn_repr
import numpy as np
import random

FILTER_MUTATIONS = np.array([16, 32, 64])



def mutate_filter(r, param):
    # TODO: implement constraints here
    new_param = random.choice(FILTER_MUTATIONS[FILTER_MUTATIONS !=r['params'][param]])
    r['params'][param] = new_param
    return r

def mutate_droprate(r,param):
    '''
    My niggas, this is the function that edits dropout rates
    This shit is so heavy no kush will make this light y'all
    
    r:= representation of the layer of the network
    param:= dropout rate to be changed.
    '''
    new_param = random.choice(FILTER_MUTATIONS[FILTER_MUTATIONS !=r['params'][param]])
    r['params'][param] = new_param
    return r
    

def mutate_random_param(reprs):
    
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
    # TODO: implement
    pass


def main():
    z=default_init_nn_repr()
    print(z)
    y = mutate_random_param(z)

if __name__ == "__main__":
    main()