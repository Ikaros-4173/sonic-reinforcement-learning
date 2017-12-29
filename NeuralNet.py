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
from PIL import Image
import SonicMain as sm
import tensorflow as tf
import ScoreReading as sr
import KeyboardEvents as ke
from collections import deque
import Quartz.CoreGraphics as CG

with open("/Users/dascienz/Desktop/sonicNN/pickles/gameover.pkl", "rb") as gameover_pickle:
    gameover = pickle.load(gameover_pickle, encoding = 'latin1')

with open("/Users/dascienz/Desktop/sonicNN/pickles/digits.pkl","rb") as digits_pickle:
    digits = pickle.load(digits_pickle, encoding='latin1')

def scoreGrab(dy):
    """Captures an image of the scoreboard and converts it to RGB array.
    Returns scoreboard readings from ScoreReading.py scoreBoard() method."""
    region = CG.CGRectMake(434,142+dy,190, 88)
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
    array[:,:,:3][other.T] = (255, 255, 255)

    return array, sr.scoreBoard(array)

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
    """Pre-process image array to (80*80) grayscale image array."""
    observation = cv2.resize(observation, (80, 80))
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_RGB2GRAY)
    return np.reshape(observation,(80, 80, 1))

def gameOverCheck(observation, dy):
    """Defines gameover condition by detecting 0 lives."""
    n = observation[828+dy:852+dy, 226:250, :3]
    return sr.compare(n, gameover)

def buttonPress(c):
    """Method for committing in-game actions. (d,a,m,s) -> (right, left, down, jump)"""
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
    
def initiateGame():
    """Initiates dy offset for score recognition."""
    sm.startState(0, reset = False)
    time.sleep(1.0)
    
    global dy
    dy = -5

    screen = scoreGrab(dy)[0]
    error = sr.compare(sr.scoreRead(screen)[0], digits[0])
    
    while error != 70.0 and dy < 5:
        screen = scoreGrab(dy)[0]
        error = sr.compare(sr.scoreRead(screen)[0], digits[0])
        dy += 1
        
    dy = (dy - 1)
    return

def testFrames():
    """Capture scores to test digit recognition."""
    def captureScore(dy):
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
        array[:,:,:3][other.T] = (255, 255, 255)
        return array
    
    initiateGame()
    
    while True:
        array = captureScore(dy=2)
        img = Image.fromarray(array)
        print(sr.scoreBoard(array))
        working_dir = '/Users/dascienz/Desktop/sonicNN/Test_Frames/'
        img.save(working_dir + 'frame_' + str(int(time.time())) + '.png', 'PNG')
    return

def getState(dy):
    """Method for getting game state and score. Score defined as rings + score."""
    state = preprocess(observationGrab())/255.0
    score = scoreGrab(dy)[1][0]+scoreGrab(dy)[1][2]
    return state, score

def trainNetwork():
    """Reinforcement learning algorithm."""
    EPISODES = 1000
    ACTIONS = 4
    GAMMA = 0.95
    EPSILON = 1.0
    BATCH = 32
    MEMORY = 25000
    STDDEV = 0.1
    LEARNING_RATE = 1e-6

    #Build the convolutional neural network.
    x = tf.placeholder(dtype=tf.float32, shape = [None, 80, 80, 1]) # input observation

    W_conv1 = tf.Variable(tf.truncated_normal(shape=[8,8,1,32], stddev=STDDEV))
    b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1,strides=[1,4,4,1],padding='SAME')+b_conv1)
    #h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[4,4,32,64], stddev=STDDEV))
    b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1,W_conv2,strides=[1,2,2,1],padding='SAME')+b_conv2)
    
    W_conv3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,64], stddev=STDDEV))
    b_conv3 = tf.Variable(tf.constant(0.01,shape=[64]))
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2,W_conv3,strides=[1,1,1,1],padding='SAME')+b_conv3)
    h_conv3_flat = tf.reshape(h_conv3,[-1,1600])
    
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[1600,512], stddev=STDDEV))
    b_fc1 = tf.Variable(tf.constant(0.01, shape=[512]))
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1)+b_fc1)
    
    W_fc2 = tf.Variable(tf.truncated_normal(shape=[512,ACTIONS], stddev=STDDEV))
    b_fc2 = tf.Variable(tf.constant(0.01, shape=[ACTIONS]))
    
    prediction = (tf.matmul(h_fc1, W_fc2) + b_fc2)

    a = tf.placeholder(dtype=tf.float32, shape=[BATCH, ACTIONS]) # action index
    y = tf.placeholder(dtype=tf.float32, shape=[BATCH]) # targets - predicted actions
    
    out_action = tf.reduce_sum(tf.multiply(prediction[0], a), axis = 1) # readout_action
    loss = tf.reduce_mean(tf.square(y - out_action))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        #checkpoint = tf.train.get_checkpoint_state("SavedNetworks")
        #if checkpoint and checkpoint.model_checkpoint_path:
            #saver.restore(sess, checkpoint.model_checkpoint_path)
        
        #Q-reinforcement algorithm for training the network.
        scores, losses = [], []
        observations = deque() #replay memory
        initiateGame() #initialize game state
        print("\n*------ GAME START ------*")
        
        for episode in range(EPISODES):
            
            if episode % 100 == 0:
                saver.save(sess, "SavedNetworks/model_%s.ckpt" % (episode))
 
            count, terminal = 0, 0
            
            done = False
            sm.loadState(0)
            print("Episode: %s" % (episode + 1) + " Random: %s" % (100.0*EPSILON))
            time.sleep(0.25)
            
            while done is False:
    
                #We are in state S
                state, score = getState(dy)
                while score < 0:
                    state, score = getState(dy) # initial state, initial score
                    count = 0
                
                #Run our Q function on S to get Q values for all possible actions.
                action = np.zeros((ACTIONS))
                if (np.random.random() <= EPSILON): #choose random action
                    i = np.random.randint(0,4)
                    action[i] = 1
                else: #choose best action from Q(s,a) values
                    qvals = prediction.eval(feed_dict={x: [state]})[0] # Qout
                    action[np.argmax(qvals)] = 1    
                action_argmax = np.argmax(action)
                buttonPress(action_argmax) #Take action.
                time.sleep(0.05)
                
                #Observe new state and score.
                new_state, new_score = getState(dy)
                while new_score < 0:
                    new_state, new_score = getState(dy)
                    count = 0

                if(gameOverCheck(observationGrab(), dy) <= 1065):
                    reward, count, terminal = -10, 0, 1
                    observations.append((state, action, reward, new_state, terminal))
                    done = True
                
                diff = new_score - score
                
                if(diff == 0):
                    count += 1
                    if(count >= 150):
                        reward, count, terminal = -1, 0, 1
                        observations.append((state, action, reward, new_state, terminal))
                        done = True
                elif(diff > 0):
                    reward, count, terminal = 10, 0, 0
                    observations.append((state, action, reward, new_state, terminal))
                elif(diff < 0):
                    reward, count, terminal = -1, 0, 1
                    observations.append((state, action, reward, new_state, terminal))
                       
            scores.append(score)
            
            #Pop first transition from replay memory.
            if len(observations) >= MEMORY:
                observations.popleft()
            
            #Replay memory loop.
            if len(observations) >= BATCH:
                batch = random.sample(observations, BATCH)
                
                state_batch = [i[0] for i in batch]
                action_batch = [i[1] for i in batch]
                reward_batch = [i[2] for i in batch]
                new_state_batch = [i[3] for i in batch]
                terminal_batch = [i[4] for i in batch]
                
                y_batch = []
                predictions = sess.run(prediction, feed_dict={x: new_state_batch})

                for idx in range(len(batch)):
                    if terminal_batch[idx] == 1: #is terminal state
                        y_batch.append(reward_batch[idx]) #reward only
                    else: #is not terminal state
                        y_batch.append(reward_batch[idx] + GAMMA*np.max(predictions[idx])) # discounted future reward
                train_step.run(feed_dict={x: state_batch, a: action_batch, y: y_batch})
                l = sess.run(loss, feed_dict={x: state_batch, a: action_batch, y: y_batch})
                print("Loss: ", l)
                losses.append(l)
            #End of replay memory loop.
            
            #Epsilon step down to decrease random actions.
            if EPSILON > 0.1:
                EPSILON -= (1/EPISODES)
        
        #End of episodes.
        sm.closeGame()
        
        #Write scores to file.
        with open("Training/scores_%s.csv" % (time.time()), "w") as outfile:
            writer = csv.writer(outfile)
            for score in scores:
                writer.writerow([score])
        
        #Write training loss to file.
        with open("Training/losses_%s.csv" % (time.time()), "w") as outfile:
            writer = csv.writer(outfile)
            for loss in losses:
                writer.writerow([loss])
        return

def testNetwork():
    with tf.Session() as sess:
        ACTIONS = 4
        STDDEV = 0.1
        
        #Build the convolutional neural network.
        x = tf.placeholder(dtype=tf.float32, shape = [None, 80, 80, 1]) # input observation

        W_conv1 = tf.Variable(tf.truncated_normal(shape=[8,8,1,32], stddev=STDDEV))
        b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1,strides=[1,4,4,1],padding='SAME')+b_conv1)
        #h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        W_conv2 = tf.Variable(tf.truncated_normal(shape=[4,4,32,64], stddev=STDDEV))
        b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1,W_conv2,strides=[1,2,2,1],padding='SAME')+b_conv2)

        W_conv3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,64], stddev=STDDEV))
        b_conv3 = tf.Variable(tf.constant(0.01,shape=[64]))
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2,W_conv3,strides=[1,1,1,1],padding='SAME')+b_conv3)
        h_conv3_flat = tf.reshape(h_conv3,[-1,1600])

        W_fc1 = tf.Variable(tf.truncated_normal(shape=[1600,512], stddev=STDDEV))
        b_fc1 = tf.Variable(tf.constant(0.01, shape=[512]))
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1)+b_fc1)

        W_fc2 = tf.Variable(tf.truncated_normal(shape=[512,ACTIONS], stddev=STDDEV))
        b_fc2 = tf.Variable(tf.constant(0.01, shape=[ACTIONS]))

        prediction = (tf.matmul(h_fc1, W_fc2) + b_fc2)

        sess.run(tf.global_variables_initializer())

        episode = 4900
        saved_graph = tf.train.import_meta_graph("SavedNetworks/model_%s.ckpt.meta" % (episode))
        saved_graph.restore(sess, "SavedNetworks/model_%s.ckpt" % (episode))
        
        initiateGame()
        time.sleep(1.5)
        play = True
        
        while play:
            
            state, score = getState(dy) # initial state, initial score
            qvals = sess.run(prediction, feed_dict={x: [state]})[0]
            action = np.argmax(qvals)
            buttonPress(action)
            
            if(gameOverCheck(observationGrab(), dy) <= 1065):
                play = False
                sm.closeGame()
