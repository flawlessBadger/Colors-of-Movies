# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:14:00 2019

@author: useuse
"""

import cv2
class movie:
    def __init__(self,path):
        self.movie = cv2.VideoCapture(path)#res/movin/bunny_sample.mp4
        success, self.frame = self.movie.read()
        self.len = int(self.movie.get(cv2.CAP_PROP_FRAME_COUNT))
        self.count = 0
    
    def move(self, frames):
        for i in range(frames):
           success, self.frame= self.movie.read()
           if(not(success)): return False
           self.count+=1
        return True
    def saveFrame(self):
        cv2.imwrite("res/frameout/frame%d.jpg" % self.count, self.frame)
        
    
    
