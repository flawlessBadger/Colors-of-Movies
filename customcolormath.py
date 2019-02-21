#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 22:26:19 2019

@author: useuse
"""

import numpy as np
import math
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from movietoframes import movie 
from skimage import io, color


# Red Color

def sat(rgb_color):
    l = lum(rgb_color)
    if(l==1 or (1-abs(2*l-1))==0): return 0
    return (np.max(rgb_color)/255-np.min(rgb_color)/255)/(1-abs(2*l-1))

def lum(rgb_color):
    return (np.max(rgb_color)/255+np.min(rgb_color)/255)/2

def vibrancy(rgb_color):
    mx = np.max(rgb_color)/255
    mn = np.min(rgb_color)/255
    df = mx-mn
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return s*v

def labchroma(rgb_color):
    return np.sqrt(np.square(rgb_color[1])+np.square(rgb_color[2]))