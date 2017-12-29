#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:08:41 2017

@author: dascienz
"""

import pickle
from scipy.linalg import norm

with open("/Users/dascienz/Desktop/sonicNN/pickles/digits.pkl","rb") as digits_pickle:
    digits = pickle.load(digits_pickle, encoding='latin1')

#score cropping
def scoreRead(array):
    """Captures digit images from the score line.
    Digit farthest to the right is considered n1."""   
    score = array[2:48, 0:379, :3]
    n1 = score[0+2:46, 348+4:379, :3]
    n2 = score[0+2:46, 316+4:347, :3]
    n3 = score[0+2:46, 284+4:315, :3]
    n4 = score[0+2:46, 252+4:283, :3]
    n5 = score[0+2:46, 220+4:251, :3]
    n6 = score[0+2:46, 188+4:219, :3]
    n7 = score[0+2:46, 156+4:187, :3]
    return n1,n2,n3,n4,n5,n6,n7
   
#timer crop
def timeRead(array):
    """Captures digit images on the timer line.
    Digit farthest to the right is considered n1."""
    time = array[66:112, 0:283, :3]
    n1 = time[0+2:46, 252+4:283, :3]
    n2 = time[0+2:46, 220+4:251, :3]
    n3 = time[0+2:46, 156+4:187, :3]
    return n1,n2,n3

#rings crop 
def ringRead(array):
    """Captures digit images on the ring line.
    Digit farthest to the right is considered n1."""
    rings = array[130:176, 0:283, :3]
    n1 = rings[0+2:46, 252+4:283, :3]
    n2 = rings[0+2:46, 220+4:251, :3]
    n3 = rings[0+2:46, 188+4:219, :3]
    return n1,n2,n3

def compare(a, b):
    diff = abs(a-b)
    return norm(diff.ravel(),0)

def digitRead(array):
    #recognize a digit from an image array 
    error = [compare(array, digits[i]) for i in range(10)]
    if (min(error) < 300.0) and (min(error) != 432.0):
        return str(error.index(min(error)))
    else:
        return str("")

def scoreBoard(array): 
    s = list(map(digitRead, scoreRead(array)))[::-1]
    t = list(map(digitRead, timeRead(array)))[::-1]
    r = list(map(digitRead, ringRead(array)))[::-1]
    try:
        score = float("".join(s).strip())
        time = (t[0]+":"+t[1]+t[2]).strip()
        rings = float("".join(r).strip())
        return (score,time,rings)
    except ValueError:
        return (-1.0, -1.0, -1.0)

def gameOverRead(observation, dy):
    n = observation[828+dy:852+dy, 226:250, :3]
    return n
