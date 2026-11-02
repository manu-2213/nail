
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from nail.sl.linear_regression import LinearRegression
from nail.performance_tests.timing import benchmark

def test_linear_regression_methods():
    """
    Performance tests to test linear regression. We start by speed, and then we test
    numerical stability
    """

    sample_sizes = np.array([100, 1000, 10_000])
    feature_sizes = np.array([5, 10, 25, 75])

    s, f = np.meshgrid(sample_sizes, feature_sizes)
    s = s.reshape(sample_sizes.shape[0] * feature_sizes.shape[0])
    f = f.reshape(sample_sizes.shape[0] * feature_sizes.shape[0])

    for i in range(len(s)):

        print(f"SAMPLE SIZE: {s[i]}, FEATURE SIZE: {f[i]}")

        X = np.random.randn(s[i], f[i])
        y = np.random.randn(s[i], 1)

        methods = ['naive', 'gram-schmidt', 'qr']

        for m in methods:
            lr = LinearRegression(X, y)
            results = benchmark(lr, m, timing="ms")
            

            print(
                f"Solving linear regression via:\n"
                f"{m} -- mean: {results['mean']:.4f} ms, -- std: {results['std']:.4f} ms\n"
                    )

    return True
        

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    test_linear_regression_methods()

