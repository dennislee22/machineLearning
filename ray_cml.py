!pip3 install pandas ray[tune] tensorboard gpustat aiohttp aiohttp_cors pydantic opencensus async_timeout
!ray start --head --block --include-dashboard=true --dashboard-port=8090 --num-gpus=1

import os
import cdsw
import ray
import time

DASHBOARD_PORT = os.environ['CDSW_APP_PORT']
DASHBOARD_IP = os.environ['CDSW_IP_ADDRESS']

ray_head_addr = DASHBOARD_IP + ':6379'
ray_head_addr

ray_url = f"ray://{DASHBOARD_IP}:10001" 
ray.init(ray_url)

# Creating 2 workers in addition to the Head Node
num_workers=2

# We need to start the ray process with --block else the command completes and the CML Worker terminates
worker_start_cmd = f"!ray start --block --address={ray_head_addr}"
#worker_start_cmd = f"!bash"

ray_workers = cdsw.launch_workers(
    n=num_workers, 
    cpu=2, 
    memory=4, 
    code=worker_start_cmd,
)

ray_worker_details = cdsw.await_workers(
    ray_workers, 
    wait_for_completion=False)
