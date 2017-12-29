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
    """Slot should be a number 0 - 9"""
    ke.KeyPress(str(slot))
    ke.KeyPress("o")
    return

def loadState(slot):
    ke.KeyPress(str(slot))
    ke.KeyPress("p")
    return

def startState(slot, reset=False):
    """Run the ROMFILE"""
    if reset == False:
        wdir = "/Users/dascienz/Desktop/dgen-sdl-1.33/"
        rom = "SonictheHedgehog.bin"
        subprocess.Popen(["./dgen", "-s", str(slot), rom], cwd=wdir)
    else:
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
    return
    
def closeGame():
    try:
        ke.KeyPress('esc')
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    return
  
def trainingMain():
    #runs the training program
    NN.trainNetwork()
    return
    
def testingMain():
    #runs the testing program
    NN.testNetwork()
    return

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
