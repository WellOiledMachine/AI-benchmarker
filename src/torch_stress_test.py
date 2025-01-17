import time
import argparse
import torch
from MonitorSpawner import MonitorSpawner


def matrix_multiply(size, device):
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    # a = a.to(device)
    # b = b.to(device)

    start_time = time.time()

    c = torch.matmul(a, b)

    torch.cuda.synchronize() if device == "cuda" else None  # Wait for computation to finish

    end_time = time.time()

    print(f"Matrix multiplication of size {size} took {end_time - start_time} seconds.")
    return c


if __name__ == "__main__":
    # Parse command-line arguments
    argparser = argparse.ArgumentParser(
        description="Run a heavy computation task with CPU or GPU."
    )
    argparser.add_argument(
        "mode", type=str, default="cpu", help="Computation mode: 'cpu' or 'cuda'."
    )
    argparser.add_argument(
        "-o", "--output", type=str, help="The directory to save the monitoring results."
    )
    args = argparser.parse_args()

    # Check if the mode is valid
    if args.mode != "cpu" and args.mode != "cuda":
        raise ValueError("Invalid mode. Please use 'cpu' or 'cuda'.")

    # Initialize the monitor
    monitor = MonitorSpawner(interval=1, output_dir=args.output)
    monitor.start_monitor()

    print(f"Running heavy computation in {args.mode} mode...")

    # Set up Matrix Sizes
    sizes = [21000]  # Allow for testing multiple matrix sizes. Matrices are square.
    iterations = 3  # Number of matrix multiplications to perform.
    # This is just so we can watch the monitoring for longer.
    for size in sizes:
        for _ in range(iterations):
            res = matrix_multiply(size, args.mode)
