import psutil
import os
import sys
tmp = int(sys.argv[1])
print("Process ID: " + str(tmp) + " currently using CPU number:" + str(psutil.Process(tmp).cpu_num()))
print("Total threads: %s" %(psutil.Process(tmp).num_threads()))
