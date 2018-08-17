#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.datasets import cifar10, fashion_mnist
from pprint import pprint
import keras
import numpy as np
import representations

(fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = fashion_mnist.load_data()
fashion_x_train = np.expand_dims(fashion_x_train, -1)
fashion_x_test = np.expand_dims(fashion_x_test, -1)

(cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = cifar10.load_data()

num_classes=10
fashion_y_train = keras.utils.to_categorical(fashion_y_train, num_classes)
fashion_y_test = keras.utils.to_categorical(fashion_y_test, num_classes)
cifar10_y_train = keras.utils.to_categorical(cifar10_y_train, num_classes)
cifar10_y_test = keras.utils.to_categorical(cifar10_y_test, num_classes)


def evaluate_nn(reprs, epochs=20, dataset='fashion', batch_size=512,
                verbose=False):
    assert dataset in ['fashion', 'cifar10']
    fitness = 0
    if dataset == 'fashion':
        x_train = fashion_x_train
        y_train = fashion_y_train
        x_test = fashion_x_test
        y_test = fashion_y_test
        shape = (28, 28, 1)
    elif dataset == 'cifar10':
        x_train = cifar10_x_train
        y_train = cifar10_y_train
        x_test = cifar10_x_test
        y_test = cifar10_y_test
        shape = (32, 32, 3)
    pprint(reprs)
    try:
        model = representations.reprs2nn(reprs, shape)
        model = keras.utils.multi_gpu_model(model)
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

        result = model.fit(x_train, y_train, batch_size=batch_size,
                           epochs=epochs, validation_data=(x_test, y_test),
                           verbose=verbose)
        fitness = max(result.history['val_acc'])
    except Exception as exc:
        print('Error evaluating...')
        print(exc)
        fitness = 0

    print(fitness)
    return fitness
