# project: AI Benchmarking
# Date: 2024-05-09

# import csv
# from traceback import print_exception
import subprocess

# import time
# import psutil
import shlex
import os
import sys


class MonitorSpawner:
    def __init__(self, output_dir, interval=None, pid=None):
        """
        Creates a MonitorSpawner object to start a monitoring process, which will
        monitor the system utilization of the python process that creates this object.

        In other words, use this class to monitor the system utilization of your python code.

        Parameters
        ----------
        output_dir : str
            The path to the directory where the monitoring results will be saved.
        interval : int, optional
            The interval (in seconds) between monitoring samples. Default is 1.
        pid : int, optional
            The process ID of the process to monitor. Default is None, which will
            monitor the process that creates this object.
            Use this to monitor a different process.
        """

        self.interval = max(1, interval)  # Ensure the interval is at least 1 second
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        print(f"Monitoring results will be saved to {self.output_dir}")

        self.parent_pid = os.getpid()
        print(f"Parent PID: {self.parent_pid}")

    def start_monitor(self):
        monitor_script = os.path.join(os.path.dirname(__file__), "monitor.py")
        command = f"{sys.executable} {monitor_script} -p {self.parent_pid} -i {self.interval} -o {self.output_dir}"

        monitor_process = subprocess.Popen(
            shlex.split(command),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"Monitoring process started with PID: {monitor_process.pid}")
