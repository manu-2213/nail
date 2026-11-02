import time
import numpy as np


def benchmark(lr_class, method, repeats: int = 20, timing: str = "ns"):
    timings = np.empty(repeats, dtype=np.int64) # time stored as nd

    for i in range(repeats):
        start = time.perf_counter_ns()
        lr_class.solve(method)
        timings[i] = time.perf_counter_ns() - start

    if timing == "ns":
        results =  {
            "mean": np.mean(timings),
            "std": np.std(timings),
        }
    elif timing == "micro":
        results =  {
            "mean": np.mean(timings) / 1e3,
            "std": np.std(timings / 1e3, correction=1),
        }
    elif timing == "ms":
        results =  {
            "mean": np.mean(timings) / 1e6,
            "std": np.std(timings / 1e6, correction=1),
        }
    elif timing == "s":
        results =  {
            "mean": np.mean(timings) / 1e9,
            "std": np.std(timings / 1e9, correction=1),
        }
    else:
          raise ValueError(f"The timing value used is not supported")
    return results
    