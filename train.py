#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.neural_net import TwoLayerNet

input_size =4
hidden_size = 10
num_classes = 3
num_inputs =5


def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size,hidden_size,num_classes,std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs,input_size)
    y = np.array([0,1,2,2,1])
    return X,y


net = init_toy_model()
X, y = init_toy_data()

stats = net.train(X,y,X,y, learning_rate=1e-1,reg=1e-5,num_iters=100,verbose=False)

print 'Final training loss:', stats['loss_history'][-1]


