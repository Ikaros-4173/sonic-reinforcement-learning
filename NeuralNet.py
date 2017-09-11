#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:50:20 2017

@author: dascienz
"""

"""
Screen resolution of 2880x1800.
(15.4-inch (2880 x 1800) retina display)
"""

import os
os.chdir("/Users/dascienz/Desktop/sonicNN/")

import cv2
import csv
import time
import pickle
import random
import numpy as np
import SonicMain as sm
import tensorflow as tf
import ScoreReading as sr
import KeyboardEvents as ke
from collections import deque
import Quartz.CoreGraphics as CG

with open("/Users/dascienz/Desktop/sonicNN/gameover.pkl", "rb") as gameover_pickle:
    gameover = pickle.load(gameover_pickle, encoding = 'latin1')

def scoreGrab(dy):
    """Captures an image of the scoreboard and converts it to RGB array.
    Returns scoreboard readings from ScoreReading.py scoreBoard() method."""
    region = CG.CGRectMake(434,142+dy,190,88)
    image = CG.CGWindowListCreateImage(region,
                                       CG.kCGWindowListOptionOnScreenOnly,
                                       CG.kCGNullWindowID,
                                       CG.kCGWindowImageDefault)
    width = CG.CGImageGetWidth(image)
    height = CG.CGImageGetHeight(image)
    bytesperrow = CG.CGImageGetBytesPerRow(image)
    pixeldata = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(image))
    
    array = np.frombuffer(pixeldata, dtype=np.uint8)
    array = array.reshape((height,bytesperrow//4,4))
    array = array[:,:,[2,1,0]][:,:width,:]
    
    r, g, b = array.T
 
    grey = (r == 196) & (g == 196) & (b == 196)
    array[:,:,:3][grey.T] = (0,0,0)
    
    other = (r > 0) & (g >= 0) & (b > 0)
    array[:,:,:3][other.T] = (255,255,255)

    return sr.scoreBoard(array,0,0,0)

def scoreArray(dy):
    """Captures an image of the scoreboard and converts it to RGB array.
    Returns scoreboard readings from ScoreReading.py scoreBoard() method."""
    region = CG.CGRectMake(434,142+dy,190,88)
    image = CG.CGWindowListCreateImage(region,
                                       CG.kCGWindowListOptionOnScreenOnly,
                                       CG.kCGNullWindowID,
                                       CG.kCGWindowImageDefault)
    width = CG.CGImageGetWidth(image)
    height = CG.CGImageGetHeight(image)
    bytesperrow = CG.CGImageGetBytesPerRow(image)
    pixeldata = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(image))
    
    array = np.frombuffer(pixeldata, dtype=np.uint8)
    array = array.reshape((height,bytesperrow//4,4))
    array = array[:,:,[2,1,0]][:,:width,:]
    
    r, g, b = array.T
 
    grey = (r == 196) & (g == 196) & (b == 196)
    array[:,:,:3][grey.T] = (0,0,0)
    
    other = (r > 0) & (g >= 0) & (b > 0)
    array[:,:,:3][other.T] = (255,255,255)

    return array

def observationGrab():
    """Captures image of the game screen and converts it to RGB array."""
    region = CG.CGRectMake(401,130,634,445)
    image = CG.CGWindowListCreateImage(region,
                                   CG.kCGWindowListOptionOnScreenOnly,
                                   CG.kCGNullWindowID,
                                   CG.kCGWindowImageDefault)
    width = CG.CGImageGetWidth(image)
    height = CG.CGImageGetHeight(image)
    bytesperrow = CG.CGImageGetBytesPerRow(image)
    pixeldata = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(image))
    array = np.frombuffer(pixeldata, dtype=np.uint8)
    array = array.reshape((height,bytesperrow//4,4))
    array = array[:,:,[2,1,0]][:,:width,:]
    return array

def preprocess(observation):
    """Pre-process image array to (57*80, 50*70, 50*50) grayscale image array."""
    observation = cv2.resize(observation, (80, 80))
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_RGB2GRAY)
    return np.reshape(observation,(80, 80))

def gameOverCheck(observation, dy):
    """Defines gameover condition by detecting 0 lives."""
    n = observation[828+dy:852+dy, 226:250, :3]
    return sr.compare(n, gameover)

def buttonPress(c):
    if c == 0:
        ke.KeyUp('a')
        return ke.KeyDown('d')
    elif c == 1:
        ke.KeyUp('d')
        return ke.KeyDown('a')
    elif c == 2:
        ke.KeyUp('s')
        return ke.KeyDown('m')
    else:
        ke.KeyUp('m')
        return ke.KeyDown('s')

def getState():
    return preprocess(observationGrab()), scoreGrab(0)[0]+scoreGrab(0)[2]

def trainNetwork():

    x = tf.placeholder(dtype=tf.float32, shape = [None, 80, 80, 4]) # input observations

    W_conv1 = tf.Variable(tf.truncated_normal(shape=[8,8,4,32],stddev=0.01))
    b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1,strides=[1,4,4,1],padding='SAME')+b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[4,4,32,64],stddev=0.01))
    b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,2,2,1],padding='SAME')+b_conv2)
    
    W_conv3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,64],stddev=0.01))
    b_conv3 = tf.Variable(tf.constant(0.01,shape=[64]))
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2,W_conv3,strides=[1,1,1,1],padding='SAME')+b_conv3)
    h_conv3_flat = tf.reshape(h_conv3,[-1,1600])
    
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[1600,512],stddev=0.01))
    b_fc1 = tf.Variable(tf.constant(0.01, shape=[512]))
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1)+b_fc1)
    
    W_fc2 = tf.Variable(tf.truncated_normal(shape=[512,4],stddev=0.01))
    b_fc2 = tf.Variable(tf.constant(0.01, shape=[4]))
    prediction = tf.matmul(h_fc1,W_fc2) + b_fc2

    a = tf.placeholder(dtype=tf.float32, shape=[None, 4]) # action indexes
    y = tf.placeholder(dtype=tf.float32, shape=[None]) # targets
    
    out_action = tf.reduce_sum(tf.multiply(prediction, a), axis = 1) # readout_action
    cost = tf.reduce_mean(tf.square(y - out_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        #checkpoint = tf.train.get_checkpoint_state("SavedNetworks")
        #if checkpoint and checkpoint.model_checkpoint_path:
            #saver.restore(sess, checkpoint.model_checkpoint_path)
        
        #Q-reinforcement algorithm for training the network.
        episodes = 1000
        gamma = 0.99
        epsilon = 1.0
    
        scores = []
        observations = deque()
        sm.startState(0)
        
        for episode in range(episodes):
            
            terminal = 0
            count = 0
            
            for step in range(5000): #While episode in progress...
                if gameOverCheck(observationGrab(), 0) <= 10:
                    sm.loadState(0)
                
                if count > 300:
                    sm.loadState(0)
                    count=0
                
                #We are in state S
                screen, score = getState() # initial state, initial score
                #state = np.reshape(screen, (80,80,1))
                ret, x_t = cv2.threshold(screen,1,255,cv2.THRESH_BINARY)
                state = np.stack((x_t, x_t, x_t, x_t), axis=2)
    
                #Run our Q function on S to get Q values for all possible actions.
                action = np.zeros((4))
                if (np.random.random() < epsilon): #choose random action
                    i = np.random.randint(0,4)
                    action[i] = 1
                else: #choose best action from Q(s,a) values
                    qvals = sess.run(prediction, feed_dict={x: [state]})[0] # Qout
                    action[np.argmax(qvals)] = 1    
                action_argmax = np.argmax(action)
                buttonPress(action_argmax) #Take action.
                
                #Observe new state and score.
                new_screen, new_score = getState()
                #new_state = np.reshape(new_screen, (80,80,1))
                ret, x_t1 = cv2.threshold(new_screen, 1, 255, cv2.THRESH_BINARY)
                x_t1 = np.reshape(x_t1, (80, 80, 1))
                new_state = np.append(x_t1, state[:, :, :3], axis=2)
                
                reward = new_score - score
                
                if reward != 0:
                    terminal = 1
                    count = 0
                else:
                    terminal = 0
                    count += 1
                
                observations.append((state, action, reward, new_state, terminal))
                
                if len(observations) > 10000:
                    observations.popleft()
                
                if len(observations) >= 5000:
                    batch = random.sample(observations, 32)
                    
                    state_batch = [d[0] for d in batch]
                    action_batch = [d[1] for d in batch]
                    reward_batch = [d[2] for d in batch]
                    new_state_batch = [d[3] for d in batch]
                    terminal_batch = [d[4] for d in batch]
                    
                    y_batch = []
                    predictions = sess.run(prediction, feed_dict={x: new_state_batch})

                    for idx in range(len(batch)):
                        if terminal_batch[idx] == 1: # is terminal state
                            y_batch.append(5*reward_batch[idx]) # reward only
                        else: # isn't terminal state
                            y_batch.append(reward_batch[idx]-10 + gamma*np.max(predictions[idx])) # discounted future reward
                    train_step.run(feed_dict={x: state_batch, a: action_batch, y: y_batch})

            print("Episode: %s" % (episode + 1) + " Epsilon: %s" % epsilon)
            scores.append(score)
            
            if episode % 100 == 0:
                saver.save(sess, "SavedNetworks/model_%s.ckpt" % (episode))
            
            if epsilon > 0.1:
                epsilon -= (1/episodes)
        
        sm.closeGame()
                
        with open('scores_%s.csv' %(time.time()), 'w') as f:
            writer = csv.writer(f)
            for score in scores:
                writer.writerow([score])
        

def testNetwork():
    with tf.Session() as sess:
        
        x = tf.placeholder(dtype=tf.float32, shape = [None, 80, 80, 4]) # input observations

        W_conv1 = tf.Variable(tf.truncated_normal(shape=[8,8,4,32],stddev=0.01))
        b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1,strides=[1,4,4,1],padding='SAME')+b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
        W_conv2 = tf.Variable(tf.truncated_normal(shape=[4,4,32,64],stddev=0.01))
        b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,2,2,1],padding='SAME')+b_conv2)
    
        W_conv3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,64],stddev=0.01))
        b_conv3 = tf.Variable(tf.constant(0.01,shape=[64]))
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2,W_conv3,strides=[1,1,1,1],padding='SAME')+b_conv3)
        h_conv3_flat = tf.reshape(h_conv3,[-1,1600])
    
        W_fc1 = tf.Variable(tf.truncated_normal(shape=[1600,512],stddev=0.01))
        b_fc1 = tf.Variable(tf.constant(0.01, shape=[512]))
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1)+b_fc1)
        
        W_fc2 = tf.Variable(tf.truncated_normal(shape=[512,4],stddev=0.01))
        b_fc2 = tf.Variable(tf.constant(0.01, shape=[4]))
        prediction = tf.matmul(h_fc1,W_fc2) + b_fc2
        
        sess.run(tf.global_variables_initializer())

        episode = 900
        saved_graph = tf.train.import_meta_graph("SavedNetworks/model_%s.ckpt.meta" % (episode))
        saved_graph.restore(sess, "SavedNetworks/model_%s.ckpt" % (episode))
        
        sm.startState(0)

        i = 0
        while i == 0:
            
            screen, score = getState() # initial state, initial score
            ret, x_t = cv2.threshold(screen,1,255,cv2.THRESH_BINARY)
            state = np.stack((x_t, x_t, x_t, x_t), axis=2)
            
            qvals = sess.run(prediction, feed_dict={x: [state]})[0]
            action = np.argmax(qvals)
            buttonPress(action)
            
            if gameOverCheck(observationGrab(), 0) <= 10:
                i = 1
                sm.closeGame()