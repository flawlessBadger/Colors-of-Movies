"""
Created on Sun Feb 17 16:14:00 2019

@author: useuse
"""


import numpy as np
import cv2
from skimage import io
from datetime import datetime
import matplotlib.pyplot as plt
from movietoframes import movie 

"""
(A) If L < 1  |  S = (Max(RGB) — Min(RGB)) / (1 — |2L - 1|)
(B) If L = 1  |  S = 0"""
def sat(*rgb_color):
    l = lum(rgb_color)
    if(l==1): return 0
    return (np.max(rgb_color)/255-np.min(rgb_color)/255)/(1-abs(2*l-1))
def lum(*rgb_color):
    return (np.max(rgb_color)/255+np.min(rgb_color)/255)/2



#img = io.imread('https://i.stack.imgur.com/DNM65.png')[:, :, :-1]
def domColor(img, plot):
    print ("start: " +  datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    #probably useless
    average = img.mean(axis=0).mean(axis=0)
    
    pixels = np.float32(img.reshape(-1, 3))
    np.float32
    
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
        
    #image create
    if(plot):
        avg_patch = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(average)
        
        indices = np.argsort(counts)[::-1]   
        freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
        rows = np.int_(img.shape[0]*freqs)
        
        dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
        
        for i in range(len(rows) - 1):
            dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10,5))
        ax0.imshow(avg_patch)
        ax0.set_title('Average color')
        ax0.axis('off')
        ax1.imshow(dom_patch)
        ax1.set_title('Dominant colors')
        ax1.axis('off')
        plt.show(fig)
        
        fig2, (ax2) = plt.subplots(1, 1, figsize=(10,5))
        ax2.imshow(dom_patch)
        ax2.set_title('color')
        ax2.axis('off')
        plt.show(fig2)
    
    print (average)
    
    #print (avg_patch)
    
    print ("finito: " +datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    return  palette[np.argmax(counts)] #dominant

    
mov = movie("res/movin/bunny_sample.mp4")#738 frames
start = datetime.now()
print ("start: " +datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


dom_patch = np.ones((739,591,3), dtype=np.uint8)
i = 0
while True:
    #print(mov.frame)
    #mov.saveFrame()
    #domColor(mov.frame, False)
    print(i)
    dom_patch[i]=dom_patch[i]*np.uint8(domColor(mov.frame, False))
    i+=1
    if(not(mov.move(int(mov.len/738)))):break
end = datetime.now()
print ("finito: " +datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
fig2, (ax2) = plt.subplots(1, 1, figsize=(50,100))
ax2.imshow(dom_patch)
ax2.set_title('wierd rabbit test')
ax2.axis('off')
plt.show(fig2)
printed = datetime.now()
img = io.imread('res/img/image-206.jpg')#[:, :, :-1]
#domColor(img, True)
#for i in range(len(rows) - 1):
  #  dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
        