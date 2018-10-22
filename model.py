#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:12:28 2018

@author: guille
"""
class network:
    def __init__(self,is_target=True,input_tensor,q_tensor=None,action_space,scope,hidden_units,L2_lambda=None,learning_rate=None):
        assert((is_target and q_tensor==None and L2_lambda==None and learning_rate==None)or(is_target==False and q_tensor!=None and L2_lambda!=None and learning_rate!=None)),"Argument mismatch, net was declared target but extra parameters were provided or was declared main and insufficient parameters were provided"
        with tf.variable_scope(scope):
            h1=tf.layers.dense(input_tensor,hidden_units,activation=tf.nn.relu,name="layer1")
            h2=tf.layers.dense(h1,hidden_units,activation=tf.nn.relu,name="layer2")
            h3=tf.layers.dense(h2,hidden_units,activation=tf.nn.relu,name="layer3")
            output=tf.layers.dense(h3,action_space,activation=None,name="output_layer")           
        weights=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=(scope))
        if is_target:
            return output,weights
        else:
            lossL2=tf.add_n([tf.nn.l2_loss(v) for v in weights if 'bias' not in v.name ])*L2_lambda
            loss=tf.add(tf.reduce_mean(tf.square(tf.subtract(output,target_q_tensor))),lossL2)
            train=tf.train.AdamOptimizer(learning_rate).minimize(loss)
            return output,weights,loss,train
