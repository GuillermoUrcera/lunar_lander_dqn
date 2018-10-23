#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import csv
import time
import replayMemory
import parameters

# Environment
env=parameters.env
STATE_SIZE=parameters.STATE_SIZE
ACTION_SPACE=parameters.ACTION_SPACE
ACTION_SIZE=parameters.ACTION_SIZE

# Hyper
MINIBATCH_SIZE=parameters.MINIBATCH_SIZE
LEARNING_RATE=parameters.LEARNING_RATE
TAU=parameters.TAU
MEMORY_MAX_SIZE=parameters.MEMORY_MAX_SIZE
L2_LAMBDA=parameters.L2_LAMBDA
DISCOUNT_FACTOR=parameters.DISCOUNT_FACTOR

# Program parameters
NUM_EPISODES=parameters.NUM_EPISODES
RENDER=parameters.RENDER
EPISODE_CHECKPOINT=parameters.EPISODE_CHECKPOINT
TIME_CHECKPOINT=parameters.TIME_CHECKPOINT
WARMUP_EPISODES=parameters.WARMUP_EPISODES

# Misc parameters
LOGS_PATH=parameters.LOGS_PATH
SAVE_PATH=parameters.SAVE_PATH
INDEX_STATE=parameters.INDEX_STATE
INDEX_REWARD=parameters.INDEX_REWARD
INDEX_DONE=parameters.INDEX_DONE
INDEX_LAST_STATE=parameters.INDEX_LAST_STATE
INDEX_ACTION=parameters.INDEX_ACTION
VAR_SIZE_DIC=parameters.VAR_SIZE_DIC

# Tuning_parameters
TUNING=parameters.TUNING
TUNER_PATH=parameters.TUNER_PATH
MIN_LEARNING_RATE=parameters.MIN_LEARNING_RATE
MAX_LEARNING_RATE=parameters.MAX_LEARNING_RATE
MIN_TAU=parameters.MIN_TAU
MAX_TAU=parameters.MAX_TAU
MAX_L2_LAMBDA=parameters.MAX_L2_LAMBDA
MIN_L2_LAMBDA=parameters.MIN_L2_LAMBDA
MIN_LR_DECAY=parameters.MIN_LR_DECAY
MAX_LR_DECAY=parameters.MAX_LR_DECAY
MINIBATCH_SIZE_LIST=parameters.MINIBATCH_SIZE_LIST
HIDDEN_UNITS_LIST=parameters.HIDDEN_UNITS_LIST

TUNING_ITERATIONS=1+(parameters.TUNING_ITERATIONS-1)*TUNING

def linear_scale(num_elements,min_value,max_value):
    scaled_list=(max_value-min_value)*np.random.random_sample(num_elements)+min_value
    return scaled_list

def log_scale(num_elements,min_value,max_value):
    scaled_list=linear_scale(num_elements,min_value,max_value)
    scaled_list=np.power(10,scaled_list)
    return scaled_list

def discrete_scale(num_elements,selection_list):
    scaled_list=np.random.randint(0,len(selection_list),size=num_elements)
    scaled_list=[selection_list[i] for i in scaled_list]
    return scaled_list

# Prepare hyperparameters
if TUNING:
    with open(TUNER_PATH,'a') as csvfile:
        spamwriter=csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Learning_rate', 'LR_decay', 'Tau','L2_reg','Minibatch_size','Hidden_units','Episodes_until_solved','Time_until_solved'])
    learning_rate_list=log_scale(TUNING_ITERATIONS,MIN_LEARNING_RATE,MAX_LEARNING_RATE)
    lr_decay_list=linear_scale(TUNING_ITERATIONS,MIN_LR_DECAY,MAX_LR_DECAY)
    tau_list=log_scale(TUNING_ITERATIONS,MIN_TAU,MAX_TAU)
    l2_list=log_scale(TUNING_ITERATIONS,MIN_L2_LAMBDA,MAX_L2_LAMBDA)
    minibatch_size_list=discrete_scale(TUNING_ITERATIONS,MINIBATCH_SIZE_LIST)
    hidden_units_list=discrete_scale(TUNING_ITERATIONS,HIDDEN_UNITS_LIST)

for tuning_iteration in range(TUNING_ITERATIONS): #TUNING_ITERATIONS = 1 if there is no tuning
    LEARNING_RATE=learning_rate_list[tuning_iteration]
    LEARNING_RATE_DECAY=lr_decay_list[tuning_iteration]
    TAU=tau_list[tuning_iteration]
    L2_LAMBDA=l2_list[tuning_iteration]
    MINIBATCH_SIZE=minibatch_size_list[tuning_iteration]
    HIDDEN_UNITS=hidden_units_list[tuning_iteration]
    
    # Init variables
    last_average=0
    epsilon=1
    learning_has_started=False
    frame=0
    done=False
    acc_Q=0
    acc_loss=0
    time_to_solution=-1
    episodes_to_solution=-1
    
    tf.reset_default_graph()
    sess=tf.Session()
    
    replayMemory=replayMemory.ReplayMemory(MINIBATCH_SIZE,MEMORY_MAX_SIZE,VAR_SIZE_DIC)
    
    state_input_tensor=tf.placeholder(tf.float32, shape=(None,STATE_SIZE),name="state_input_tensor")
    target_Q_tensor=tf.placeholder(tf.float32, shape=(None,ACTION_SPACE),name="Q_tensor")
    
    with tf.variable_scope("main_network"):
        h1=tf.layers.dense(state_input_tensor,HIDDEN_UNITS,activation=tf.nn.relu,name="layer1")
        h2=tf.layers.dense(h1,HIDDEN_UNITS,activation=tf.nn.relu,name="layer2")
        h3=tf.layers.dense(h2,HIDDEN_UNITS,activation=tf.nn.relu,name="layer3")
        output=tf.layers.dense(h3,ACTION_SPACE,activation=None,name="output_layer")
    main_weights=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=("main_network"))
    lossL2=tf.add_n([tf.nn.l2_loss(v) for v in main_weights if 'bias' not in v.name ])*L2_LAMBDA
    loss=tf.add(tf.reduce_mean(tf.square(tf.subtract(output,target_Q_tensor))),lossL2)
    train=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    
    with tf.variable_scope("target_network"):
        h1_target=tf.layers.dense(state_input_tensor,HIDDEN_UNITS,activation=tf.nn.relu,name="layer1")
        h2_target=tf.layers.dense(h1_target,HIDDEN_UNITS,activation=tf.nn.relu,name="layer2")
        h3_target=tf.layers.dense(h2_target,HIDDEN_UNITS,activation=tf.nn.relu,name="layer3")
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
    time0=time.time()
    for episode in range(NUM_EPISODES):  
        if episode%EPISODE_CHECKPOINT==0:
            print "Episode",episode,"of",NUM_EPISODES
            if episode%TIME_CHECKPOINT==0:
                time1=time.time()
                elapsed_time=time1-time0
                h=int(elapsed_time/3600)
                m=int((elapsed_time%3600)/60)
                s=elapsed_time%60
                print "Elapsed time:",h,"h",m,",m",s,"s"
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
                if TUNING:
                    episodes_to_solution=episode
                    time_to_solution=time.time()-t0
            # Store transition
            replayMemory.add(new_state,reward,done,state,action)
            state=new_state
            if episode>WARMUP_EPISODES:
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
                    if episode%100==0:
                        # Check if rewards have stagnated and reduce learning rate
                        if last_average!=0:
                            if (moving_average-last_average<(0.01*moving_average)) and (LEARNIG_RATE>0.001):
                                LEARNING_RATE*=LEARNING_RATE_DECAY
                        last_average=moving_average
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
    if TUNING:
        with open(TUNER_PATH,'a') as csvfile:
            spamwriter=csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([LEARNING_RATE,LEARNING_RATE_DECAY,TAU,L2_LAMBDA,MINIBATCH_SIZE,HIDDEN_UNITS,episodes_to_solution,time_to_solution])
        print "Tuning results recorded at: ",TUNER_PATH
