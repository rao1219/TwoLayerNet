#!/usr/bin/env python
# coding=utf-8
import numpy as np

# f = 1 / 1 + exp(-wx+b)

class backward_f(object):
    def __init__(self,w,x,b):
        self.params = {}
        self.params['w'] = w
        self.params['x'] = x
        self.params['b'] = b

    def backward(self):
        grads = {}
        w = self.params['w']
        x = self.params['x']
        b = self.params['b']
        
        # 前向传播
        exp_s = -1*w*x+b
        exp = np.exp(exp_s)
        row = 1+ exp
        f = 1.0/row

        # 反向传播
        df = 1.0
        drow = -1.0/row*row *df
        dexp = drow*1.0
        dexp_s = np.exp(exp_s) * dexp
        dw = -1*x*dexp_s
        dx = -1*w*dexp_s
        db = dexp_s
        
        grads['w'] = dw
        grads['x'] = dx
        grads['b'] = db

        return f,grads



fun = backward_f(1,2,3)

f,grads = fun.backward()
print 'f,',f 
print grads['w'],grads['x'],grads['b']
