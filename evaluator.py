#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 20:16:53 2018

@author: guille
"""
import tensorflow as tf
import gym

# Enviroment parameters
env = gym.make('LunarLander-v2')
STATE_SIZE=env.observation_space.shape[0]
ACTION_SPACE=env.action_space.n
ACTION_SIZE=1

# Evaluator parameters:
EPISODES_TO_EVALUATE=20
EPOCHS_PER_EPISODE_TO_EVALUATE=500
VISUALIZE_EVALUATOR=True
SAVE_PATH="tmp/lander_weights"

# Program start
tf.reset_default_graph()
sess=tf.Session()

# Recreate network
state_input_tensor=tf.placeholder(tf.float32, shape=(None,STATE_SIZE),name="state_input_tensor")

with tf.variable_scope("main_network"):
    h1=tf.layers.dense(state_input_tensor,256,activation=tf.nn.relu,name="layer1")
    h2=tf.layers.dense(h1,256,activation=tf.nn.relu,name="layer2")
    h3=tf.layers.dense(h2,256,activation=tf.nn.relu,name="layer3")
    output=tf.layers.dense(h3,ACTION_SPACE,activation=None,name="output_layer")

saver = tf.train.Saver()

tf.get_default_graph().finalize()

# Restore variables from disk.
saver.restore(sess,SAVE_PATH)

# Evaluate model
for e in range(EPISODES_TO_EVALUATE):
	state=env.reset()
	for i in range(EPOCHS_PER_EPISODE_TO_EVALUATE):
		if VISUALIZE_EVALUATOR:
			env.render()
		state,reward,done, _=env.step(sess.run(output,feed_dict={state_input_tensor:[state]}))
	env.render(close=True)