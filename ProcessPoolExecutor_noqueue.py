from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import threading
import os
import time
import random
import numpy as np

open('output', 'w').close()    
with open("readme",'r') as fp:
  x = len(fp.readlines())

def createList(r1, r2):
    return np.arange(r1, r2)
  
z = createList(1, x)
  
def write_to_file(lineno):    
    with open("output", "a+") as file:
        file.write("current threads:" + str(threading.current_thread()))
        file.write("\n")
        file.write("current pid:" + str(os.getpid()))
        file.write("\n")
        file.write("total threads:" + str(psutil.Process().num_threads()))
        file.write("\n")
        file.write("CPU number:" + str(psutil.Process().cpu_num()))
        file.write("\n")
        sleeptime = random.randint(1, 10)
        #time.sleep(sleeptime)
        with open("readme",'r') as ok:
          global y
          #y += 0
          content = ok.readlines()
          file.write("line " + content[lineno])
          #print(sleeptime, lineno)
          
        file.write("\n")
        file.close()

if __name__ == "__main__":
    start = time.time()
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = executor.map(write_to_file,z)
    end = time.time()
    print("Time Taken with Multiprocessing:{}".format(end - start))
