import pynvml
import psutil
import time
import argparse
import csv
import os
from datetime import datetime
import subprocess
from collections import namedtuple

MONITOR_PID = os.getpid()  # The PID of the monitor process

# Define a named tuple datatype to make the code more readable
ProcessInfo = namedtuple(
    "ProcessInfo",
    [
        "cpu_percent",
        "ram_used",
        "threads",
        "disk_read_bytes",
        "disk_write_bytes",
        "disk_read_ops",
        "disk_write_ops",
    ],
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Monitor the system utilization of a process."
    )
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "-p",
        "--pid",
        type=int,
        help="The PID of the process to monitor.",
    )
    group.add_argument(
        "-c",
        "--command",
        nargs=argparse.REMAINDER,
        help="The command to run the process to monitor.",
    )

    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=1,
        help="The interval (in seconds) between monitoring samples.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "monitoring_results"),
        help="The directory to save the monitoring results.",
    )
    args = parser.parse_args()
    # Expand user and resolve any symlinks in the path
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    return args


def sync_process_children(process, children_map={}):
    """
    Update the dictionary of child processes of a given process.
    This function will take care not to redefine any of the existing process
    objects in the dictionary. Doing so would cause the process objects to
    reset their CPU and memory usage statistics every time they we wanted to update them.

    Parameters
    ----------
    process : psutil.Process
        The parent process whose children will be updated.
    children_map : dict, optional
        The dictionary of child processes to update.
        Omit this parameter to create a new dictionary, with new process objects.

    Returns
    -------
    children_map : dict
        The updated (or newly created) dictionary of child processes.
    """
    global MONITOR_PID

    current_children = set(process.children(recursive=True))

    # Remove children that are no longer running
    for pid in list(children_map.keys()):
        # Make sure that the monitor process is never monitored
        # Also, remove the child if it is no longer a child of the parent process
        if pid == MONITOR_PID or not any(
            child.pid == pid for child in current_children
        ):
            del children_map[pid]

    # Add new children
    for child in current_children:
        if child.pid != MONITOR_PID and child.pid not in children_map:
            children_map[child.pid] = child

    return children_map


def get_ps_info(process):
    """
    Get the CPU, RAM, and Disk utilization of a process.

    Parameters
    ----------
    process : psutil.Process
        The process to query.

    Returns
    -------
    ProcessInfo: namedtuple
        A namedtuple containing the CPU, RAM, and Disk utilization of the process.
    """
    try:
        cpu_percent = process.cpu_percent(interval=None)
        ram_used = process.memory_info().rss  # in bytes
        threads = process.num_threads()

        disk_io = process.io_counters()
        disk_read_bytes = disk_io.read_bytes
        disk_write_bytes = disk_io.write_bytes
        disk_read_ops = disk_io.read_count
        disk_write_ops = disk_io.write_count
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None

    return ProcessInfo(
        cpu_percent,
        ram_used,
        threads,
        disk_read_bytes,
        disk_write_bytes,
        disk_read_ops,
        disk_write_ops,
    )


if __name__ == "__main__":
    args = parse_arguments()

    # Grab the timestamp to append to the output file paths
    monitor_timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    # Initialize NVML to monitor GPU stats
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()

    args.output_dir = os.path.join(args.output_dir, monitor_timestamp)
    os.makedirs(args.output_dir, exist_ok=True)

    # Define output files. One for process info and one for GPU info
    psinfo_file = os.path.join(args.output_dir, "psinfo.csv")
    gpuinfo_file = os.path.join(args.output_dir, "gpuinfo.csv")

    # Open output files and monitor the process
    with (
        open(psinfo_file, mode="w", newline="") as psinfo,
        open(gpuinfo_file, mode="w", newline="") as gpuinfo,
    ):
        ps_writer = csv.writer(psinfo)
        ps_writer.writerow(
            [
                "Timestamp (Unix)",
                "CPU Utilization (%)",
                "Thread Count",
                "RAM Utilization (Bytes)",
                "RAM Utilization (%)",
                "Disk Read (Bytes)",
                "Disk Write (Bytes)",
                "Disk Read Operations (Count)",
                "Disk Write Operations (Count)",
            ]
        )

        gpu_writer = csv.writer(gpuinfo)
        gpu_writer.writerow(
            [
                "Timestamp (Unix)",
                "GPU ID",
                "GPU Utilization (%)",
                "GPU Memory Used (B)",
                "GPU Memory Percent (%)",
                "GPU Temperature (°C)",
                "Power Usage (W)",
            ]
        )

        # If a command is provided, run the command and monitor the process
        if args.command:
            # Convert list to single string. Required for using shell=True in Popen
            command = " ".join(args.command)

            # Execute command and redirect stdout and stderr to log files
            stdout = os.path.join(args.output_dir, "stdout.log")
            stderr = os.path.join(args.output_dir, "stderr.log")
            with open(stdout, "w") as out, open(stderr, "w") as err:
                # print(f"Running command: {command}")
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                )
            parent_proc = psutil.Process(proc.pid)
        else:
            parent_proc = psutil.Process(args.pid)

        child_proc_map = sync_process_children(parent_proc)
        total_ram_avail = psutil.virtual_memory().total

        while parent_proc.is_running() and parent_proc.status() != psutil.STATUS_ZOMBIE:
            start_timestamp = time.time()

            # Get Process Info from parent
            parent_info = get_ps_info(parent_proc)
            if parent_info is None:
                break

            # Extract info from dictionary to make code more readable
            cpu_percent = parent_info.cpu_percent
            total_ram_used = parent_info.ram_used
            threads = parent_info.threads
            disk_read_bytes = parent_info.disk_read_bytes
            disk_write_bytes = parent_info.disk_write_bytes
            disk_read_ops = parent_info.disk_read_ops
            disk_write_ops = parent_info.disk_write_ops

            # Aggregate CPU and RAM usage for child processes
            for child in child_proc_map:
                child_info = get_ps_info(child_proc_map[child])
                if child_info is not None:
                    cpu_percent += child_info.cpu_percent
                    total_ram_used += child_info.ram_used
                    threads += child_info.threads
                    disk_read_bytes += child_info.disk_read_bytes
                    disk_write_bytes += child_info.disk_write_bytes
                    disk_read_ops += child_info.disk_read_ops
                    disk_write_ops += child_info.disk_write_ops

            total_ram_used_percent = (total_ram_used / total_ram_avail) * 100

            ps_writer.writerow(
                [
                    start_timestamp,
                    cpu_percent,
                    threads,
                    total_ram_used,
                    total_ram_used_percent,
                    disk_read_bytes,
                    disk_write_bytes,
                    disk_read_ops,
                    disk_write_ops,
                ]
            )

            for i in range(gpu_count):
                gpu = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu)
                memory = pynvml.nvmlDeviceGetMemoryInfo(gpu)
                temperature = pynvml.nvmlDeviceGetTemperature(gpu, 0)
                power = pynvml.nvmlDeviceGetPowerUsage(gpu) / 1000
                mem_percent = memory.used / memory.total * 100
                gpu_writer.writerow(
                    [
                        start_timestamp,
                        i,
                        utilization.gpu,
                        memory.used,
                        mem_percent,
                        temperature,
                        power,
                    ]
                )

            # Update the child process map
            child_proc_map = sync_process_children(parent_proc, child_proc_map)

            # Sleep for the remaining time in the interval
            end_timestamp = time.time()
            elapsed_time = end_timestamp - start_timestamp
            sleep_time = max(0, args.interval - elapsed_time)
            time.sleep(sleep_time)

        ps_writer.writerow(["FINISHED"])
        gpu_writer.writerow(["FINISHED"])
    pynvml.nvmlShutdown()
