#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:14:44 2017

@author: guille
"""
import collections
import random
import numpy as np

class replayMemory:
    def __init__(self,minibatch_size,max_size,size_dic):
        self.memory=collections.deque(maxlen=max_size)
        self.minibatch_size=minibatch_size
        self.var_dic=size_dic
    def add(self, state, reward, done, last_state, action):
        self.memory.append((state, reward, done, last_state, action))
    def get_batch(self):
        batch_size = min(self.minibatch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        return minibatch
    def get(self,index,buff):
        return np.asarray([col[index] for col in buff])
    def get_from_minibatch(self,minibatch,index):
        batch_size=len(minibatch)
        var_size=self.var_dic.get(index)
        return np.asarray(self.get(index,minibatch)).reshape(batch_size,var_size)