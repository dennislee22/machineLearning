import psutil
import threading
import os
import time

#lockme = threading.Lock()

open('output', 'w').close()    

y = 0
def write_to_file():
    with open("readme",'r') as fp:
      x = len(fp.readlines())
    #while lockme.locked():
    #  continue
    #lockme.acquire()
    with open("output", "a+") as file:
      file.write("current threads:" + str(threading.current_thread()))
      file.write("\n")
      file.write("current pid:" + str(os.getpid()))
      file.write("\n")
      file.write("total threads:" + str(psutil.Process().num_threads()))
      file.write("\n")
      file.write("CPU number:" + str(psutil.Process().cpu_num()))
      file.write("\n")
      with open("readme",'r') as ok:
        global y
        y -= 1 
        content = ok.readlines()
        file.write("line " + content[y])
        print(y)
      file.write("\n")
      file.close()

    #lockme.release()

threads = []

start_time = time.perf_counter()

for i in range(1, 21):
    t = threading.Thread(target=write_to_file)
    threads.append(t)
    t.start()
[thread.join() for thread in threads]

end_time = time.perf_counter()  
execution_time = end_time - start_time  
print(f"\nJob Starts: {start_time}\nJob Ends: {end_time}\nTotals Execution Time:{execution_time:0.2f} seconds.")


