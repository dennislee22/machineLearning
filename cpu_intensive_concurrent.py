from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import cv2
import numpy as np
import psutil
import os
import sys

original_stdout = sys.stdout
open('yyy', 'w').close()    


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

def track_cpu_usage():
    cpu_prec_usage = psutil.cpu_percent(interval=1)
    cpunum_int = int (float(psutil.cpu_percent(interval=1)))
    print ("Process id", os.getpid(),"\"",psutil.Process().name(),"\" is running on CPU core: ",psutil.Process().cpu_num())
 
def compute_intensive_function(image_in):
    # Resizing
    image_in = dict_images[image_in]
    smaller_image = cv2.resize(image_in, (100, 100), interpolation=1)

    # Rotation
    rows, cols = image_in.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    dst_rotation = cv2.warpAffine(image_in, M, (cols, rows))

    # Translation
    M = np.float32([[1, 0, -100], [0, 1, -100]])
    dst_translation = cv2.warpAffine(image_in, M, (cols, rows))

    # Edge detection
    edges = cv2.Canny(image_in, 100, 200)

    # Kernel
    averaging_kernel = np.ones((3, 3), np.float32) / 9
    filtered_image = cv2.filter2D(image_in, -1, averaging_kernel)

    # Gaussian Kernel
    gaussian_kernel_x = cv2.getGaussianKernel(5, 1)
    gaussian_kernel_y = cv2.getGaussianKernel(5, 1)
    gaussian_kernel = gaussian_kernel_x * gaussian_kernel_y.T
    filtered_image = cv2.filter2D(image_in, -1, gaussian_kernel)

    # Image contours
    gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
    # calculate the contours from binary image
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    with_contours = cv2.drawContours(image_in, contours, -1, (0, 255, 0), 3)
    original_stdout = sys.stdout  
    with open('yyy', 'a') as f:
          sys.stdout = f
          print(psutil.Process().cpu_num())
          sys.stdout = original_stdout  
    return True
     

            
if __name__ == "__main__":
    start = time.time()
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = executor.map(
            compute_intensive_function,
            list_images * 10,         
        )

    end = time.time()
    print("Time Taken with Multiprocessing:{}".format(end - start))
