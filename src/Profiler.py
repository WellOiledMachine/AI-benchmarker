# Authors: Tyson Limato, Cody Sloan
# project: AI Benchmarking
# Date: 2024-05-09

import csv
import subprocess
import shlex
from traceback import print_exception
import multiprocessing
import time
import psutil
import os
import pynvml


class monitor_system_utilization:
    def __init__(self, interval=None, output_dir=None, pid=os.getpid(), update_iteration=100):
        self.interval = max(1, interval)  # Ensure the interval is at least 1 second
        self.output_dir = output_dir
        self.update_iteration = max(1, update_iteration)  # Ensure the interval is at least 1 iteration
        self.cpu_ram_file_path = os.path.join(self.output_dir, "CPU_RAM_Utilization.csv")
        self.monitor_process = None
        self.running = True  # Flag to control the monitoring loop
        self.gpustat_file_path = os.path.join(self.output_dir, "GPU_Utilization.csv")
        self.parent_pid = pid
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure the directory exists
        

    def start_monitoring(self):
        self.monitor_process = multiprocessing.Process(target=self.monitor_system_utilization_helper)
        self.running = multiprocessing.Value('b', True)
        self.monitor_process.start()
    
    def stop_monitoring(self):
        self.__exit__(None, None, None)

    def monitor_system_utilization_helper(self):
        monitor_pid = os.getpid()  # Get the PID of the monitoring process
        
        # Initialize NVML to monitor GPU stats
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        # Get the parent process and all child processes
        parent_process = psutil.Process(self.parent_pid)
        child_processes = [p for p in parent_process.children(recursive=True) if p.pid != monitor_pid]



        with open(self.cpu_ram_file_path, mode='w', newline='') as results_file, \
             open(self.gpustat_file_path, mode='w', newline='') as gpu_file:
            cpu_writer = csv.writer(results_file)
            cpu_writer.writerow(['Timestamp (Unix)', 'CPU Utilization (%)', 'Thread Count', 'RAM Utilization (%)', 'RAM Utilization (MB)'])

            gpu_writer = csv.writer(gpu_file)
            gpu_writer.writerow(['Timestamp (Unix)', 'GPU ID','GPU Utilization (%)', 'GPU Memory Used (MB)', \
                            'GPU Memory Total (MB)', 'GPU Temperature (°C)', 'Power Usage (W)' ])
            
            # Interval for updating the child processes
            update_counter = 0
            while self.running.value:
                iteration_start_time = time.time()
                update_counter += 1
                # Aggregate CPU and RAM usage for child processes
                cpu_percent = parent_process.cpu_percent(interval=None)
                total_ram_used = parent_process.memory_info().rss  # in bytes
                threads = parent_process.num_threads()
                for child in child_processes:
                    cpu_percent += child.cpu_percent(interval=None)
                    total_ram_used += child.memory_info().rss  # in bytes
                    threads += child.num_threads()

                ram_mb = total_ram_used / 1048576  # Convert bytes to megabytes (MB)
                ram_percent = (total_ram_used / psutil.virtual_memory().total) * 100
                # print(cpu_percent)
                cpu_writer.writerow([time.time(), cpu_percent, threads, ram_percent, ram_mb])

                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    timestamp = time.time()
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle) 
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                    gpu_writer.writerow([timestamp, i, utilization.gpu, memory.used / 1048576, memory.total / 1048576, temperature, power_usage])

                if update_counter >= self.update_iteration:
                    child_processes = [p for p in parent_process.children(recursive=True) if p.pid != monitor_pid]
                    for child in child_processes:
                        child.cpu_percent(interval=None)
                    update_counter = 0

                remaining_iteration_time = self.interval - (time.time() - iteration_start_time)
                if remaining_iteration_time > 0:
                    time.sleep(remaining_iteration_time)


    def __enter__(self):
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        print("Monitor stopped.")
        self.running.value = False
        # if hasattr(self, 'gpu_stat_process') and self.gpu_stat_process:
        #     self.gpu_stat_process.terminate()
        #     self.gpu_stat_process.wait()
        if hasattr(self, 'monitor_process') and self.monitor_process:
            self.monitor_process.join()
        if exc_type is not None:
            print_exception(exc_type, exc_value, traceback)
