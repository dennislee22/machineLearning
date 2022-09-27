import os
import cdsw
import ray

DASHBOARD_PORT = os.environ['CDSW_APP_PORT']

# We need to start the ray process with --block else the command completes and the CML Worker terminates
head_start_cmd = f"!pip3 install pandas ray[tune] tensorboard gpustat aiohttp aiohttp_cors pydantic opencensus async_timeout; ray start --head --block --include-dashboard=true --dashboard-port={DASHBOARD_PORT}"
ray_head = cdsw.launch_workers(
    n=1,
    cpu=2,
    memory=16,
    code=head_start_cmd,
)

ray_head_details = cdsw.await_workers(
  ray_head, 
  wait_for_completion=False, 
  timeout_seconds=90
)

ray_head_ip = ray_head_details['workers'][0]['ip_address']
ray_head_addr = ray_head_ip + ':6379'
ray_head_addr

# Creating 2 workers in addition to the Head Node
num_workers=2

# We need to start the ray process with --block else the command completes and the CML Worker terminates
worker_start_cmd = f"!ray start --block --address={ray_head_addr}"

ray_workers = cdsw.launch_workers(
    n=num_workers, 
    cpu=2, 
    memory=4, 
    code=worker_start_cmd,
)

ray_worker_details = cdsw.await_workers(
    ray_workers, 
    wait_for_completion=False)

dashboard_url = ray_head_details['workers'][0]['app_url']
dashboard_url

ray_url = f"ray://{ray_head_ip}:10001" 
ray.init(ray_url)
