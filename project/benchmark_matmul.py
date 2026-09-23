"""Compare a Python loop with FastOps and, if available, CUDA."""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
from numba import config, cuda, get_num_threads

import minitorch


def naive(a, b):
    n = len(a)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            value = 0.0
            for k in range(n):
                value += a[i][k] * b[k][j]
            out[i][j] = value
    return out


def measure(fn, repeats, gpu=False):
    times = []
    for _ in range(repeats):
        if gpu:
            cuda.synchronize()
        start = time.perf_counter()
        fn()
        if gpu:
            cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in [("python", "Python: три цикла"), ("cpu", "Numba CPU"), ("gpu", "CUDA GPU")]:
        if all(key in row for row in rows):
            ax.plot([r["size"] for r in rows], [r[key] * 1000 for r in rows], "o-", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Размер квадратной матрицы")
    ax.set_ylabel("Время, мс (логарифмическая шкала)")
    ax.set_title("Матричное умножение: медиана трёх запусков")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/matmul_benchmark.json")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    args = parser.parse_args()
    random.seed(42)
    backends = {"cpu": minitorch.TensorBackend(minitorch.FastOps)}
    if cuda.is_available() and not config.ENABLE_CUDASIM:
        backends["gpu"] = minitorch.TensorBackend(minitorch.CudaOps)
    rows = []
    for n in args.sizes:
        a = [[random.random() for _ in range(n)] for _ in range(n)]
        b = [[random.random() for _ in range(n)] for _ in range(n)]
        expected = np.asarray(a) @ np.asarray(b)
        np.testing.assert_allclose(naive(a, b), expected)
        row = {"size": n, "python": measure(lambda: naive(a, b), 3)}
        for name, backend in backends.items():
            x = minitorch.tensor(a, backend=backend)
            y = minitorch.tensor(b, backend=backend)
            # Warm up compilation and check the result before timing.
            result = (x @ y)._tensor._storage
            if name == "gpu":
                result = result.copy_to_host()
            np.testing.assert_allclose(result.reshape(n, n), expected)
            row[name] = measure(lambda: x @ y, 3, gpu=name == "gpu")
        rows.append(row)
        print(row, flush=True)
    Path(args.output).write_text(json.dumps({"threads": get_num_threads(), "results": rows}, indent=2) + "\n")
    if args.plot:
        plot(rows, Path(args.output).with_suffix(".png"))
