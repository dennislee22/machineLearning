import threading
import time
import psutil
import os
import concurrent.futures
import shutil
import sys

original_stdout = sys.stdout
open('zzz', 'w').close()    

def write_file(mythread):
   with open('zzz', 'a') as f:
      sys.stdout = f
      print("Task",mythread)
      sys.stdout = original_stdout

def thread1():
    print("thread start 1")
    time.sleep(30)
    write_file(thread1.__name__)
    print("Thread 1 Process id", os.getpid(),"\"",psutil.Process().name(),"\" is running on CPU core: ",psutil.Process().cpu_num())
    print(threading.current_thread())
    print("total threads:",psutil.Process().num_threads())
    print("thread finished 1")

def thread2():
    print("thread start 2")
    time.sleep(25)
    write_file(thread2.__name__)
    print ("Thread 2 Process id", os.getpid(),"\"",psutil.Process().name(),"\" is running on CPU core: ",psutil.Process().cpu_num())
    print(threading.current_thread())
    print("total threads:",psutil.Process().num_threads())
    print("thread finished 2")

def thread3():
    print("thread start 3")
    time.sleep(20)
    write_file(thread3.__name__)
    print ("Thread 3 Process id", os.getpid(),"\"",psutil.Process().name(),"\" is running on CPU core: ",psutil.Process().cpu_num())
    print(threading.current_thread())
    print("total threads:",psutil.Process().num_threads())
    print("thread finished 3")
    
def thread4():
    print("thread start 4")
    time.sleep(15)
    write_file(thread4.__name__)
    print ("Thread 4 Process id", os.getpid(),"\"",psutil.Process().name(),"\" is running on CPU core: ",psutil.Process().cpu_num())
    print(threading.current_thread())
    print("total threads:",psutil.Process().num_threads())
    print("thread finished 4")
    
def thread5():
    print("thread start 5")
    time.sleep(10)
    write_file(thread5.__name__)
    print ("Thread 5 Process id", os.getpid(),"\"",psutil.Process().name(),"\" is running on CPU core: ",psutil.Process().cpu_num())
    print(threading.current_thread())
    print("total threads:",psutil.Process().num_threads())
    print("thread finished 5")
    
def thread6():
    print("thread start 6")
    #shutil.copy('a.iso', 'b.iso')
    time.sleep(5)
    write_file(thread6.__name__)
    print ("Thread 6 Process id", os.getpid(),"\"",psutil.Process().name(),"\" is running on CPU core: ",psutil.Process().cpu_num())
    print(threading.current_thread())  
    print("total threads:",psutil.Process().num_threads())  
    print("thread finished 6")    

def main():
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    f1 = executor.submit(thread1)
    f2 = executor.submit(thread2)
    f3 = executor.submit(thread3)
    f4 = executor.submit(thread4)
    f5 = executor.submit(thread5)
    f6 = executor.submit(thread6)
start_time = time.perf_counter()
main()


print("process exited")
end_time = time.perf_counter()  
execution_time = end_time - start_time  
print(f"\nJob Starts: {start_time}\nJob Ends: {end_time}\nTotals Execution Time:{execution_time:0.2f} seconds.")
with open('zzz', 'r') as f:
    print(f.read())
