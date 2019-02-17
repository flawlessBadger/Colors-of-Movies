# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:14:00 2019

@author: useuse
"""

import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('res/movin/bunny_sample.mp4')
eachxframes = 10
success,image = vidcap.read()
count = 0
success = True
while success:
  if(count%10==0):
      cv2.imwrite("res/frameout/frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print ('Read a new frame: '+ str(success))
  count += 1