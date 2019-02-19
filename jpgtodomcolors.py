"""
Created on Sun Feb 17 16:14:00 2019

@author: useuse
"""


import numpy as np
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from movietoframes import movie 

"""
(A) If L < 1  |  S = (Max(RGB) — Min(RGB)) / (1 — |2L - 1|)
(B) If L = 1  |  S = 0"""
TOTALFRAMES = 20
save = False
n_colors = 3
mov = movie("res/movin/The.Fall.aka.Pad.2006.1080p.BluRay.H264.AAC-RARBG.mp4")#738 frames

def sat(*rgb_color):
    l = lum(rgb_color)
    if(l==1 or (1-abs(2*l-1))==0): return 0
    return (np.max(rgb_color)/255-np.min(rgb_color)/255)/(1-abs(2*l-1))

def lum(*rgb_color):
    return (np.max(rgb_color)/255+np.min(rgb_color)/255)/2

def vibrance(*rgb_color):
    l = lum(rgb_color)
    if(l<0.5): l=l*2
    else:  l=(1-l)*2
    return sat(rgb_color)*l

#img = io.imread('https://i.stack.imgur.com/DNM65.png')[:, :, :-1]

def domColor(img, plot, debug):
    print ("start: " +  datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    #probably useless
    b,g,r = cv2.split(img)
    img = cv2.merge((r,g,b))
    
    average = img.mean(axis=0).mean(axis=0)
    
    pixels = np.float32(img.reshape(-1, 3))
    
    #n_colors = 5
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
        

    if debug:
        print ("avg color: "+str(average))
        print ("dom color: "+str(palette[np.argmax(counts)]))
        print ("palette: "+str(palette))
        print ("freq: "+str(counts))
    
    satdom = []
    satavg = vibrance(average)    
    for i in range(palette.shape[0]):
        satdom.append(vibrance(palette[i]))        
    maxsatcol = palette[satdom.index(max(satdom))]

    
    counts_copy= counts.copy()
    maxcount = np.argmax(counts_copy)
    while (satdom[maxcount]<satavg):
        counts_copy[maxcount] = 0
        maxcount = np.argmax(counts_copy)
        if counts_copy[maxcount]==0:
            maxcount = np.argmax(counts)
            break
    first_sat_col  = palette[maxcount]
    if debug:
        print("saturations: "+str(satdom))
        print ("most saturated: " + str(maxsatcol))
        print ("first saturated: " + str(first_sat_col))

    #np.delete(a, index)
    #print("sorted i guess: "+ str(np.array(list(zip(palette[0],counts)))))
    
    #print (avg_patch)
    if(plot):
        compare_patch = np.ones((4,4,3), dtype=np.uint8)
        compare_patch[0]=compare_patch[0]*np.uint8(average)
        compare_patch[1]=compare_patch[1]*np.uint8(palette[np.argmax(counts)])
        compare_patch[2]=compare_patch[2]*np.uint8(first_sat_col)
        compare_patch[3]=compare_patch[3]*np.uint8(maxsatcol)
    

        fig2, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,5))
       # b,g,r = cv2.split(img)
        #frame_rgb = cv2.merge((r,g,b))
        #img = frame_rgb
        ax1.imshow(img)
        ax1.set_title('image')
        ax1.axis('off')
        ax2.imshow(compare_patch)
        ax2.set_title('compare_patch')
        ax2.axis('off')
        plt.show(fig2)
    
    if debug: print ("finito: " +datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return  average, palette[np.argmax(counts)], first_sat_col, maxsatcol #most saturated
    return  
    #return palette 
    
def printgraphs(name):
    figa, (axa) = plt.subplots(1, 1, figsize=(40,50))
    axa.imshow(average_patch)
    axa.set_title('average_patch')
    axa.axis('off')
    #plt.show(figd)
    if(save):
        figa.savefig('res/average_patch'+name+".png")
    
    figb, (axb) = plt.subplots(1, 1, figsize=(40,50))
    axb.imshow(dominant_patch)
    axb.set_title('dominant_patch')
    axb.axis('off')
    #plt.show(figd)
    if(save):
        figb.savefig('res/dominant_patch'+name+".png")
    
    figc, (axc) = plt.subplots(1, 1, figsize=(40,50))
    axc.imshow(firstsat_patch)
    axc.set_title('firstsat_patch')
    axc.axis('off')
    #plt.show(figd)
    if(save):
        figc.savefig('res/firstsat_patch'+name+".png")
    
    figd, (axd) = plt.subplots(1, 1, figsize=(40,50))
    axd.imshow(topsat_patch)
    axd.set_title('topsat_patch')
    axd.axis('off')
    #plt.show(figa)
    if(save):
        figd.savefig('res/topsat_patch'+name+".png")
    printed = datetime.now()
#mov = movie("res/movin/bunny_sample.mp4")#738 frames
#domColor(img, True)
    
    
start = datetime.now()
print ("start: " +datetime.now().strftime("%Y-%m-%d %H:%M:%S"))





average_patch = np.ones((TOTALFRAMES+1,int(TOTALFRAMES*4/5),3), dtype=np.uint8)
dominant_patch = np.ones((TOTALFRAMES+1,int(TOTALFRAMES*4/5),3), dtype=np.uint8)
firstsat_patch = np.ones((TOTALFRAMES+1,int(TOTALFRAMES*4/5),3), dtype=np.uint8)
topsat_patch = np.ones((TOTALFRAMES+1,int(TOTALFRAMES*4/5),3), dtype=np.uint8)
i = 0
pos = 0

while True:
    average, dominant, firstsat, topsat = domColor(mov.frame, True, False)
    average_patch[i]=average_patch[i]*np.uint8(average)
    dominant_patch[i]=dominant_patch[i]*np.uint8(dominant)
    firstsat_patch[i]=firstsat_patch[i]*np.uint8(firstsat)
    topsat_patch[i]=topsat_patch[i]*np.uint8(topsat)
    i+=1
    pos+= mov.len/TOTALFRAMES
    print("index: "+str(i)+"/"+str(TOTALFRAMES)+" frame: "+str(pos))
    if(not(mov.move(int(pos)))): break



end = datetime.now()
print ("finito: " +datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
printgraphs("3c")



#domColor(img, True)
#for i in range(len(rows) - 1):
  #  dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
        