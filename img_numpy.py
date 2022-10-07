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
ori_img_array = asarray(img)

def img_to_numpy():  
    img = Image.open('me.png')
    plt.imshow(img)
    print("Image format:", img.format)
    print("Image size:", img.size)
    print("Image mode:", img.mode)
    numpyraw = asarray(img)  
    #np.set_printoptions(threshold=sys.maxsize)
    np.save('bbb.npy',numpyraw)
    print("Values of original image:\n",numpyraw[0:1,0:3])  
    
def numpy_to_img():  
    numpyraw = np.load('bbb.npy')
    print("Some contents of numpy file:\n",numpyraw[0:1,0:3])
    plt.imshow(numpyraw)   

def tamper_img_fast():
    imgtmp = img
    width, height = imgtmp.size
    pix = img.load()
    for x in range(0, width):
        for y in range(0, height):
            if pix[x,y] == old_color:
                imgtmp.putpixel((x, y), new_color)
    plt.imshow(imgtmp)
    tampered_img_array = asarray(imgtmp)
    print("Values of original image:\n",ori_img_array[0:1,0:3])    
    print("\nValues of tampered image:\n",tampered_img_array[0:1,0:3])

def tamper_numpy_slow():
    numpyraw = np.load('bbb.npy')
    for x in range(0, width):
        for y in range(0, height):
            if (numpyraw[x][y] == [0, 0, 0, 0]).all() :
                numpyraw[x][y] = [5,3,192,255]
    print("Values of original image:\n",ori_img_array[0:1,0:3])    
    print("\nValues of tampered image:\n",numpyraw[0:1,0:3])
    plt.imshow(numpyraw)

def cv2_multithread_npy():
    edges = cv2.Canny(ori_img_array, 100, 200)
    plt.imshow(edges)
    
start = time.time()
for image in range(1):
  #img_to_numpy()
  #numpy_to_img()
  #tamper_img_fast()
  #tamper_numpy_slow()
  cv2_multithread_npy()
end = time.time()
print("Time Taken:{}".format(end - start))
