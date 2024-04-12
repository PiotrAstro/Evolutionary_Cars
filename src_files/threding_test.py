import os
import time
from concurrent.futures import ThreadPoolExecutor

os.environ["PYTHON_GIL"] = "0"

def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

time_start = time.perf_counter()
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [
        executor.submit(fib, 30)
        for _ in range(5)
    ]
    results = [future.result() for future in futures]
time_end = time.perf_counter()
print(f"Time: {time_end - time_start} with 5 threads")

time_start = time.perf_counter()
results = [fib(30) for _ in range(5)]
time_end = time.perf_counter()
print(f"Time: {time_end - time_start} with 1 thread")