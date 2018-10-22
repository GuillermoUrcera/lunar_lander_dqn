#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import gym
import numpy as np
import tensorflow as tf
import replayMemory

env = gym.make('LunarLander-v2')

STATE_SIZE=env.observation_space.shape[0]
ACTION_SPACE=env.action_space.n
ACTION_SIZE=1

MINIBATCH_SIZE=32
LEARNING_RATE=1e-4 
TAU=1e-4
MEMORY_MAX_SIZE=1e6
L2_LAMBDA=1e-3
DISCOUNT_FACTOR=0.99

NUM_EPISODES=5000
RENDER=False
EPISODE_CHECKPOINT=25

LOGS_PATH="/tmp/lander_logs"
SAVE_PATH="tmp/lander_weights"
INDEX_STATE=0
INDEX_REWARD=1
INDEX_DONE=2
INDEX_LAST_STATE=3
INDEX_ACTION=4
VAR_SIZE_DIC={INDEX_STATE:STATE_SIZE,
              INDEX_REWARD:1,
              INDEX_DONE:1,
              INDEX_LAST_STATE:STATE_SIZE,
              INDEX_ACTION:ACTION_SIZE}

epsilon=1
learning_has_started=False
warmup_episodes=500
frame=0
done=False
acc_Q=0
acc_loss=0


tf.reset_default_graph()
sess=tf.Session()

replayMemory=replayMemory.replayMemory(MINIBATCH_SIZE,MEMORY_MAX_SIZE,VAR_SIZE_DIC)

state_input_tensor=tf.placeholder(tf.float32, shape=(None,STATE_SIZE),name="state_input_tensor")
target_Q_tensor=tf.placeholder(tf.float32, shape=(None,ACTION_SPACE),name="Q_tensor")

with tf.variable_scope("main_network"):
    h1=tf.layers.dense(state_input_tensor,256,activation=tf.nn.relu,name="layer1")
    h2=tf.layers.dense(h1,256,activation=tf.nn.relu,name="layer2")
    h3=tf.layers.dense(h2,256,activation=tf.nn.relu,name="layer3")
    output=tf.layers.dense(h3,ACTION_SPACE,activation=None,name="output_layer")
main_weights=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=("main_network"))
lossL2=tf.add_n([tf.nn.l2_loss(v) for v in main_weights if 'bias' not in v.name ])*L2_LAMBDA
loss=tf.add(tf.reduce_mean(tf.square(tf.subtract(output,target_Q_tensor))),lossL2)
train=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.variable_scope("target_network"):
    h1_target=tf.layers.dense(state_input_tensor,256,activation=tf.nn.relu,name="layer1")
    h2_target=tf.layers.dense(h1_target,256,activation=tf.nn.relu,name="layer2")
    h3_target=tf.layers.dense(h2_target,256,activation=tf.nn.relu,name="layer3")
    output_target=tf.layers.dense(h3_target,ACTION_SPACE,activation=None,name="output_layer")
target_weights=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=("target_network"))

update_target_ops=[]
with tf.variable_scope("target_update"):
    for i in range(len(main_weights)):
        update_target_op=target_weights[i].assign(TAU*main_weights[i]+(1-TAU)*target_weights[i])
        update_target_ops.append(update_target_op)

#Tensorboard
writer=tf.summary.FileWriter(LOGS_PATH,sess.graph)
loss_summary=tf.placeholder('float',name='loss_value')
reward_summary=tf.placeholder('float',name='reward_value')

loss_sum=tf.summary.scalar("loss", loss_summary)
re_sum=tf.summary.scalar("reward", reward_summary)

summaryMerged=tf.summary.merge_all()
saver=tf.train.Saver()
init_op=tf.global_variables_initializer()
tf.get_default_graph().finalize()
sess.run(init_op)   
    
landed=False
moving_average=0
for episode in range(NUM_EPISODES):  
    if episode%EPISODE_CHECKPOINT==0:
        print "Episode",episode,"of",NUM_EPISODES
    done=False
    state=env.reset()
    acc_reward=0
    epoch=0
    while not done:
        epoch+=1
        # Select action
        if np.random.random()<epsilon:
            # Random action
            action=np.random.randint(0,ACTION_SPACE)
        else:
            Qs=sess.run(output,feed_dict={state_input_tensor:[state]})
            action=np.argmax(Qs)
        if epsilon>0.1:
            epsilon=-9e-7*frame+1
        new_state,reward,done,_=env.step(action)
        acc_reward+=reward
        if reward==100 and not landed:
            print "First successful landing!"
            landed=True
        # Store transition
        replayMemory.add(new_state,reward,done,state,action)
        state=new_state
        if episode>warmup_episodes:
            if not learning_has_started:
                print "Warmup phase over!"
                learning_has_started=True
            frame+=1
            #Sample minibatch
            minibatch=replayMemory.get_batch()
            S=replayMemory.get_from_minibatch(minibatch,INDEX_STATE)
            St0=replayMemory.get_from_minibatch(minibatch,INDEX_LAST_STATE)
            A=replayMemory.get_from_minibatch(minibatch,INDEX_ACTION)
            D=replayMemory.get_from_minibatch(minibatch,INDEX_DONE)
            R=replayMemory.get_from_minibatch(minibatch,INDEX_REWARD)     
            # Create targets
            next_states_Q=R+DISCOUNT_FACTOR*np.reshape(np.max(sess.run(output_target,feed_dict={state_input_tensor:S}),axis=-1),(MINIBATCH_SIZE,1))*(1-D)
            target_Q=sess.run(output,feed_dict={state_input_tensor:St0})
            target_Q[np.arange(MINIBATCH_SIZE),A.flatten()]=np.transpose(next_states_Q)
            # Train network
            acc_loss+=sess.run(loss,feed_dict={state_input_tensor:St0,target_Q_tensor:target_Q})
            sess.run(train,feed_dict={state_input_tensor:St0,target_Q_tensor:target_Q})
            # Update target network
            sess.run(update_target_ops)
            if done:
                sumOut=sess.run(re_sum,feed_dict={reward_summary:acc_reward})
                writer.add_summary(sumOut,episode)
                moving_average=0.99*moving_average+0.01*acc_reward
                if moving_average>=200:
                    print "Problem solved after",episode,"episodes!"
                acc_reward=0
                # record loss
                mean_loss=float(acc_loss)/(epoch)
                summary_loss=sess.run(loss_sum,feed_dict={loss_summary:mean_loss})
                writer.add_summary(summary_loss,episode)
                acc_loss=0
saver.save(sess,SAVE_PATH)
print "Model saved in path: ",SAVE_PATH

