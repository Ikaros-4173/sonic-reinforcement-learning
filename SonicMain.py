#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:27:52 2017

@author: dascienz
"""

import os
os.chdir("/Users/dascienz/Desktop/sonicNN/")

import NeuralNet as NN
import subprocess, time
import KeyboardEvents as ke

def saveState(slot):
    # Slot should be a number 0 - 9
    ke.KeyPress(str(slot))
    ke.KeyPress("o")

def loadState(slot):
    ke.KeyPress(str(slot))
    ke.KeyPress("p")

def startState(slot):
    #run ROMFILE
    
    wdir = "/Users/dascienz/Desktop/dgen-sdl-1.33/"
    rom = "SonictheHedgehog.bin"
    subprocess.Popen(["./dgen", "-s", str(slot), rom], cwd=wdir)
    
    """
    wdir = "/Users/dascienz/Desktop/dgen-sdl-1.33/"
    rom = "SonictheHedgehog.bin"
    subprocess.Popen(["./dgen", rom], cwd=wdir)
    
    #hit start at the title screen
    time.sleep(9.25)
    ke.KeyDown('.') 
    time.sleep(0.25)
    ke.KeyUp('.')
    time.sleep(0.25)
    saveState(0)
    """    

def closeGame():
    try:
        ke.KeyPress('esc')
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass

"""
from PIL import Image
def playSonic():
    NN.runQ()
    while True:
        img = Image.fromarray(observationGrab())
        working_dir = '/Users/dascienz/Desktop/sonicNN/frames/'
        img.save(working_dir + 'frame_' + str(int(time.time())) + '.png', 'PNG')
"""

#run game  
def trainingMain():
    #runs the training program
    NN.trainNetwork()
    
def testingMain():
    #runs the testing program
    NN.testNetwork()

if __name__ == '__main__':
    prompt = input("Do you want to 'train' or 'test'?\n>>")
    while(type(prompt) == str):
        if prompt == "train":
            trainingMain()
            break
        elif prompt == "test":
            testingMain()
            break
        else:
            print("Goodbye.")
            break