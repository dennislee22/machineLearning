import psutil
import threading
import os
import time
import concurrent.futures

open('output', 'w').close()    
with open("input",'r') as fp:
  x = len(fp.readlines())
fp.close()
y = 0

def write_to_file():
    with open("output", "a+") as file:
      file.write("current thread:" + str(threading.current_thread()))
      file.write("\n")
      file.write("current pid:" + str(os.getpid()))
      file.write("\n")
      file.write("total threads:" + str(psutil.Process().num_threads()))
      file.write("\n")
      file.write("CPU number:" + str(psutil.Process().cpu_num()))
      file.write("\n")
      with open("input",'r') as ok:
        global y
        content = ok.readlines()
        file.write(content[y])
        #print(y)
        y += 1
      file.write("\n")
      file.close()

if __name__ == "__main__":
  start_time = time.perf_counter()
  with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(write_to_file) for i in range(x)]
  end_time = time.perf_counter()  
  execution_time = end_time - start_time  
  print(f"\nJob Starts: {start_time}\nJob Ends: {end_time}\nTotals Execution Time:{execution_time:0.2f} seconds.")
