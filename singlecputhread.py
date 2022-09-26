import psutil
import os
import time

open('output', 'w').close()    
with open("readme",'r') as fp:
  x = len(fp.readlines())
y = 0
def write_to_file():
    with open("output", "a+") as file:
      file.write("current threads:" + str(threading.current_thread()))
      file.write("\n")
      file.write("current pid:" + str(os.getpid()))
      file.write("\n")
      file.write("total threads:" + str(psutil.Process().num_threads()))
      file.write("\n")
      file.write("CPU number:" + str(psutil.Process().cpu_num()))
      file.write("\n")
      time.sleep(3)
      with open("readme",'r') as ok:
        global y
        y += 1
        content = ok.readlines()
        file.write("line " + content[y])
        print(y)
      file.write("\n")
      file.close()


if __name__ == "__main__":
    start_time = time.perf_counter()
    for i in range(1, 20):
      write_to_file()
    end_time = time.perf_counter() 
    execution_time = end_time - start_time  
    print(f"\nJob Starts: {start_time}\nJob Ends: {end_time}\nTotals Execution Time:{execution_time:0.2f} seconds.")
   

