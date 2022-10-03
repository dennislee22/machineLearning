import psutil
import os
import sys
import GPUtil
import time
tmp = int(sys.argv[1])
print("Process ID: " + str(tmp) + " currently using CPU number:" + str(psutil.Process(tmp).cpu_num()))
print("GPU utilization", GPUtil.showUtilization())
print("Total threads: %s" %(psutil.Process(tmp).num_threads()))
time.sleep(1)
