# monitoring/system_monitor.py

import psutil
import time
import threading
from monitoring.metrics import cpu_usage, gpu_usage

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False


def monitor_system(interval=2):
    while True:
        # CPU
        cpu_usage.set(psutil.cpu_percent())

        # GPU
        if GPU_AVAILABLE:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage.set(util.gpu)
        else:
            gpu_usage.set(0)

        time.sleep(interval)


def start_system_monitor():
    thread = threading.Thread(target=monitor_system, daemon=True)
    thread.start()
