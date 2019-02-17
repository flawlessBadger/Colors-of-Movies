"""
Created on Sun Feb 17 16:14:00 2019

@author: useuse
"""


import numpy as np
import cv2
from skimage import io
from datetime import datetime

"""
(A) If L < 1  |  S = (Max(RGB) — Min(RGB)) / (1 — |2L - 1|)
(B) If L = 1  |  S = 0"""
def sat(*rgb_color):
    #minc =  min(float(s) for s in rgb_color)/255
    #maxc =  max(float(s) for s in rgb_color)/255
    l = (np.max(rgb_color)/255+np.min(rgb_color)/255)/2
    print ("what"+str(l))
    if(l==1): return 0
    print("here?: "+str(np.min(rgb_color)))
    return (np.max(rgb_color)/255-np.min(rgb_color)/255)/(1-abs(2*l-1))


img = io.imread('res/img/image-442.jpg')#[:, :, :-1]
#img = io.imread('https://i.stack.imgur.com/DNM65.png')[:, :, :-1]
print ("fo real: " +  datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

pixels = np.float32(img.reshape(-1, 3))
np.float32
#probably useless
average = img.mean(axis=0).mean(axis=0)

n_colors = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
flags = cv2.KMEANS_RANDOM_CENTERS

_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
_, counts = np.unique(labels, return_counts=True)
dominant = palette[np.argmax(counts)]

print ("computed: " +datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
starttime = datetime.now()

#image create
import matplotlib.pyplot as plt

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
print (sat(average))

print ("finito: " +datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


"""
(A) If L < 1  |  S = (Max(RGB) — Min(RGB)) / (1 — |2L - 1|)
(B) If L = 1  |  S = 0"""