#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import representations
from keras.datasets import mnist


def evaluate_nn(reprs):
    model = representations.reprs2nn(reprs)
    model.compile('adam', 'categorical-crossentropy', metrics=['accuracy'])

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    result = model.fit(x_train, y_train, batch_size=32, epochs=100,
                       validation_data=(x_test, y_test), verbose=1)
    return max(result.history['val_acc'])
