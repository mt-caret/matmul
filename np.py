import numpy as np
import time
import sys

def numpy_matrix_multiply(m1, m2):
    return np.dot(m1, m2)

def benchmark_numpy(size, runs):
    m1 = np.random.rand(size, size)
    m2 = np.random.rand(size, size)

    # Warmup
    for _ in range(10):
        _ = numpy_matrix_multiply(m1, m2)

    print("size,run,runtime_us,method")
    
    for i in range(runs):
        start_time = time.time_ns()
        result = numpy_matrix_multiply(m1, m2)
        end_time = time.time_ns()
        runtime_us = int((end_time - start_time) / 1e3)
        print(f"{size},{i+1},{runtime_us},numpy")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python np.py <size> <runs>")
        sys.exit(1)
    
    try:
        size = int(sys.argv[1])
        runs = int(sys.argv[2])
        
        if size <= 0 or runs <= 0:
            raise ValueError("Size and runs must be positive integers.")
        
        benchmark_numpy(size, runs)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)