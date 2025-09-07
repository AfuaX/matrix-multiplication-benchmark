import time
import threading
import random
import numpy as np
import multiprocessing as mp
import timeit
import matplotlib.pyplot as plt 

def generate_matrix(n, m):
    """Generate a random n x m matrix with integers 1-10."""
    return [[random.randint(1, 10) for _ in range(m)] for _ in range(n)]

# Single-threaded matrix multiplication
def matrix_multiply(a, b):
    rows_a = len(a)
    cols_a = len(a[0])
    cols_b = len(b[0])
    c = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                c[i][j] += a[i][k] * b[k][j]
    return c

# Multi-threaded worker function
def worker_multiply(args):
    a, b, start_row, end_row, result_list, idx = args
    cols_a = len(a[0])
    cols_b = len(b[0])
    partial_c = [[0] * cols_b for _ in range(end_row - start_row)]
    for i in range(start_row, end_row):
        for j in range(cols_b):
            for k in range(cols_a):
                partial_c[i - start_row][j] += a[i][k] * b[k][j]
    result_list[idx] = (partial_c, start_row)

# Multi-threaded matrix multiplication
def multi_threaded_multiply(a, b, num_threads=4):
    rows_a = len(a)
    cols_a = len(a[0])
    cols_b = len(b[0])
    chunk_size = rows_a // num_threads
    result_list = [None] * num_threads
    threads = []
    for t in range(num_threads):
        start_row = t * chunk_size
        end_row = start_row + chunk_size if t < num_threads - 1 else rows_a
        if start_row >= rows_a:
            break
        args = (a, b, start_row, end_row, result_list, t)
        thread = threading.Thread(target=worker_multiply, args=(args,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    c = [[0] * cols_b for _ in range(rows_a)]
    for partial_c_start in result_list:
        if partial_c_start is not None:
            partial_c, start_row = partial_c_start
            for i, row in enumerate(partial_c):
                c[start_row + i] = row
    return c

# Multi-processing worker function
def worker_multiply_process(a, b, start_row, end_row):
    cols_a = len(a[0])
    cols_b = len(b[0])
    partial_c = [[0] * cols_b for _ in range(end_row - start_row)]
    for i in range(start_row, end_row):
        for j in range(cols_b):
            for k in range(cols_a):
                partial_c[i - start_row][j] += a[i][k] * b[k][j]
    return partial_c, start_row

# Multi-processing matrix multiplication
def multi_process_multiply(a, b, num_processes=4):
    rows_a = len(a)
    cols_b = len(b[0])
    chunk_size = rows_a // num_processes
    args_list = [(a, b, i * chunk_size, (i + 1) * chunk_size if i < num_processes - 1 else rows_a) for i in range(num_processes)]
    with mp.Pool(num_processes) as pool:
        results = pool.starmap(worker_multiply_process, args_list)
    c = [[0] * cols_b for _ in range(rows_a)]
    for partial_c, start_row in results:
        for i, row in enumerate(partial_c):
            c[start_row + i] = row
    return c

# NumPy-based matrix multiplication
def numpy_multiply(a_list, b_list):
    a_np = np.array(a_list)
    b_np = np.array(b_list)
    return np.dot(a_np, b_np)

# Test harness with timeit
def main():
    n = 200  # Matrix size (n x n)
    a = generate_matrix(n, n)
    b = generate_matrix(n, n)

    # Number of timeit iterations
    num_runs = 5

    # Single-threaded
    single_time = timeit.timeit(lambda: matrix_multiply(a, b), number=num_runs) / num_runs
    c1 = matrix_multiply(a, b)
    print(f"Single-threaded time (avg of {num_runs} runs): {single_time:.4f} seconds")

    # Multi-threaded
    multi_thread_time = timeit.timeit(lambda: multi_threaded_multiply(a, b, 4), number=num_runs) / num_runs
    c2 = multi_threaded_multiply(a, b, 4)
    print(f"Multi-threaded time (4 threads, avg of {num_runs} runs): {multi_thread_time:.4f} seconds")
    print(f"Results match (first row): {c1[0][:5] == c2[0][:5]}")

    # Multi-processing
    multi_process_time = timeit.timeit(lambda: multi_process_multiply(a, b, 4), number=num_runs) / num_runs
    c_mp = multi_process_multiply(a, b, 4)
    print(f"Multi-processing time (4 processes, avg of {num_runs} runs): {multi_process_time:.4f} seconds")
    print(f"Results match (first row): {c1[0][:5] == c_mp[0][:5]}")

    # NumPy
    numpy_time = timeit.timeit(lambda: numpy_multiply(a, b), number=num_runs) / num_runs
    c_np = numpy_multiply(a, b)
    print(f"NumPy time (avg of {num_runs} runs): {numpy_time:.4f} seconds")

    # --- Speedup summary table ---
    print("\n=== Performance Summary (vs Single-threaded) ===")
    print(f"{'Method':<20}{'Time (s)':<15}{'Speedup':<10}")
    print("-" * 45)
    print(f"{'Single-threaded':<20}{single_time:<15.4f}{'1.00×':<10}")
    print(f"{'Multi-threaded':<20}{multi_thread_time:<15.4f}{single_time / multi_thread_time:,.2f}×")
    print(f"{'Multi-processing':<20}{multi_process_time:<15.4f}{single_time / multi_process_time:,.2f}×")
    print(f"{'NumPy (BLAS)':<20}{numpy_time:<15.4f}{single_time / numpy_time:,.2f}×")

    # --- Plot results ---
    methods = ["Single-threaded", "Multi-threaded", "Multi-processing", "NumPy (BLAS)"]
    times = [single_time, multi_thread_time, multi_process_time, numpy_time]

    plt.figure(figsize=(8,5))
    bars = plt.bar(methods, times, color=["#4e79a7","#f28e2b","#76b7b2","#59a14f"])
    plt.ylabel("Time (seconds)")
    plt.title("Matrix Multiplication Performance (n=200, avg of 5 runs)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Label bars with values
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{t:.4f}", 
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("perf_summary.png", dpi=200)
    print("\nPlot saved to perf_summary.png")

if __name__ == "__main__":
    main()
