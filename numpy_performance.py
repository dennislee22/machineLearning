import time
import cv2
import numpy as np
from numpy import asarray
import psutil
import os
import matplotlib.pylab as plt
import sys
from PIL import Image

new_color = 5,3,192,255
old_color = 0, 0, 0, 0
width, height = 988,988
img = Image.open('me.png')
cvimg = cv2.imread('me.png')
original = asarray(img)

def compute_fast_1thread():
    #print("1st array of original image:\n",original[0:1,0:3])
    im = img
    width, height = im.size
    pix = img.load()
    for x in range(0, width):
        for y in range(0, height):
            if pix[x,y] == old_color:
                im.putpixel((x, y), new_color)
    raw = asarray(im)
    #print("\n1st array of tampered image:\n",raw[0:1,0:3])
    return True

def compute_slow_1thread():
    #print("1st array of original image:\n",original[0:1,0:3])
    numpyraw = np.load('bbb.npy')
    for x in range(0, 988):
        for y in range(0, 988):
            if (numpyraw[x][y] == [0, 0, 0, 0]).all() :
                numpyraw[x][y] = [19,195,5,255]
    #print("\n1st array of tampered image:\n",numpyraw[0:1,0:3])
    #plt.imshow(numpyraw)
    return True

def compute_cv2_multithread1process_npy():
    numpyraw = np.full((height, width, 3), 6, np.uint8)
    edges = cv2.Canny(numpyraw, 100, 200)
    #plt.imshow(edges)
    
def compute_cv2_multithread1process_iosave():
    edges = cv2.Canny(cvimg, 100, 200)
    np.save('test.npy',edges)

start = time.time()
for image in range(10000):
  compute_fast_1thread()
  #compute_slow_1thread()
  #compute_cv2_multithread1process_npy()
  #compute_cv2_multithread1process_iosave()
#plt.imshow('output.png')
end = time.time()
print("Time Taken:{}".format(end - start))
