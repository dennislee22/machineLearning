import time
import cv2
import numpy as np
from numpy import asarray
import psutil
import os
import matplotlib.pylab as plt
import sys
from PIL import Image

#np.set_printoptions(threshold=sys.maxsize)

height = 512
width = 512
dict_images = {
    "img1": np.zeros((height, width, 3), np.uint8),
    "img2": np.ones((height, width, 3), np.uint8),
    "img3": np.full((height, width, 3), 2, np.uint8),
    "img4": np.full((height, width, 3), 3, np.uint8),
    "img5": np.full((height, width, 3), 4, np.uint8),
    "img6": np.full((height, width, 3), 5, np.uint8),
    "img7": np.full((height, width, 3), 6, np.uint8),
}

list_images = list(dict_images)

  
def compute_intensive_function():
    # Resizing
    #image_in = dict_images[image_in]
    #smaller_image = cv2.resize(image_in, (100, 100), interpolation=1)

    # Rotation
    #rows, cols = image_in.shape[:2]
    #M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    #dst_rotation = cv2.warpAffine(image_in, M, (cols, rows))

    # Edge detection
    
    #img = cv2.imread('me.png',0)
    #img = Image.open('me.png')
    #plt.imshow(img)
    
    #numpyraw = asarray(img)
    numpyraw = np.load('bbb.npy')
    edges = cv2.Canny(numpyraw, 100, 200)
    #img_arr = np.array(edges)
    #print(img_arr)
    
    #new_img = Image.fromarray(img_arr)
    new_img = Image.fromarray(edges)
    #plt.imshow(new_img)
    
    #img_arr = np.array(img)
    #print(img_arr.shape)
    
    #img = np.full((height, width, 3), 6, np.uint8)
    
    #array = np.zeros([100, 200, 4], dtype=np.uint8)
    #array[:,:100] = [255, 128, 0, 255] #Orange left side
    #array[:,100:] = [0, 0, 255, 255]   #Blue right side
    
    # Set transparency based on x position
    #array[:, :, 3] = np.linspace(0, 255, 200)
    
    #edges = cv2.Canny(img, 100, 200)
    
    #numpyraw = asarray(img)
    
    #print(img.format)
    #print(img.size)
    #print(img.mode)

    #plt.imshow(img)
    #print(numpyraw)
    
    #img_arr = np.array(img)
    #print(img_arr.shape)
    
    #img_arr = img_arr - 180
    #new_img = Image.fromarray(img_arr)
    #numpyraw2 = asarray(new_img)
    
    #img = Image.open('me.png')
    #numpyraw = asarray(img)
    #plt.imshow(new_img)
    #np.save('bbb.npy',numpyraw)
    #a = np.load('bbb.npy')
    #print(a)
    return True

#original_stdout = sys.stdout
#img = Image.open('me.png')
#plt.imshow(img)

start = time.time()
for image in range(10000):
  compute_intensive_function()
  

end = time.time()
print("Time Taken:{}".format(end - start))
