#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import representations
import numpy as np
from keras.datasets import mnist
import keras

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=3) #add the 'channel' dimension, in order to use Conv2D!
x_test = np.expand_dims(x_test, axis=3)

#to categorical, otherwise last dense layer expects 1 input only:
num_classes=10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#subset of data:
x_train = x_train[:200]
y_train = y_train[:200]
x_test = x_test[:50]
y_test = y_test[:50]



def evaluate_nn(reprs, epochs=5):
    fitness = 0
    try: 
        model = representations.reprs2nn(reprs)
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])


        result = model.fit(x_train, y_train, batch_size=32, epochs=2,
                           validation_data=(x_test, y_test), verbose=1)
        fitness = max(result.history['val_acc'])
    except:
        print("Error evaluating..")

    return fitness


def evaluate_nn_test(reprs):

    model = representations.reprs2nn(reprs)
    print(model.summary())

    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    result = model.fit(x_train, y_train, batch_size=32, epochs=3,
                       validation_data=(x_test, y_test), verbose=1)
    return max(result.history['val_acc'])


