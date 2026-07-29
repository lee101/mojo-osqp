"""Honest end-to-end benchmarks against upstream osqp."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import osqp
import scipy
from scipy import sparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_osqp  # noqa: E402


SETTINGS = dict(
    verbose=False,
    eps_abs=1.0e-5,
    eps_rel=1.0e-5,
    max_iter=6000,
)


def problem(n, m, seed=0, box=False):
    rng = np.random.default_rng(seed)
    diagonal = rng.uniform(0.5, 2.0, size=n)
    if box:
        P = sparse.diags(diagonal, format="csc")
        A = sparse.eye(n, format="csc")
        q = rng.normal(size=n)
        return P, q, A, np.full(n, -0.5), np.full(n, 0.5)
    low_rank = rng.normal(size=(max(3, n // 8), n))
    P = sparse.csc_matrix(np.diag(diagonal) + low_rank.T @ low_rank / n)
    A_dense = rng.normal(size=(m, n)) / np.sqrt(n)
    feasible = rng.normal(size=n)
    center = A_dense @ feasible
    width = rng.uniform(0.3, 1.0, size=m)
    return (
        P,
        rng.normal(size=n),
        sparse.csc_matrix(A_dense),
        center - width,
        center + width,
    )


def configured(module, data):
    solver = module.OSQP()
    solver.setup(P=data[0], q=data[1], A=data[2], l=data[3], u=data[4], **SETTINGS)
    return solver


def best_time(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best


def setup_solve_case(data):
    def ours():
        return configured(mojo_osqp, data).solve()

    def theirs():
        return configured(osqp, data).solve()

    left = ours()
    right = theirs()
    if not np.allclose(left.x, right.x, atol=2.0e-4, rtol=2.0e-4):
        raise RuntimeError("benchmark solutions do not agree")
    return ours, theirs


def hot_solve_case(data):
    ours_solver = configured(mojo_osqp, data)
    their_solver = configured(osqp, data)
    ours_solver.solve()
    their_solver.solve()
    return ours_solver.solve, their_solver.solve


def update_case(data):
    ours_solver = configured(mojo_osqp, data)
    their_solver = configured(osqp, data)
    ours_solver.solve()
    their_solver.solve()
    q0 = data[1].copy()
    ours_counter = [0]
    theirs_counter = [0]

    def ours():
        ours_counter[0] += 1
        ours_solver.update(q=q0 + (ours_counter[0] % 2) * 1.0e-3)
        return ours_solver.solve()

    def theirs():
        theirs_counter[0] += 1
        their_solver.update(q=q0 + (theirs_counter[0] % 2) * 1.0e-3)
        return their_solver.solve()

    return ours, theirs


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    cases = [
        ("dense setup + solve (32 vars, 64 cons)", setup_solve_case(problem(32, 64))),
        ("dense setup + solve (96 vars, 192 cons)", setup_solve_case(problem(96, 192, 1))),
        ("box setup + solve (256 vars)", setup_solve_case(problem(256, 256, 2, box=True))),
        ("warm solve (96 vars, 192 cons)", hot_solve_case(problem(96, 192, 3))),
        ("q update + warm solve (96 vars, 192 cons)", update_case(problem(96, 192, 4))),
    ]

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print(
        f"Versions: mojo-osqp {mojo_osqp.__version__}; upstream osqp "
        f"{osqp.__version__}; NumPy {np.__version__}; SciPy {scipy.__version__}"
    )
    print("Method: best of 5 wall-clock runs per implementation and case")
    print()
    print("| case | mojo-osqp | upstream osqp | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, (ours, theirs) in cases:
        mojo_time = best_time(ours)
        upstream_time = best_time(theirs)
        ratio = upstream_time / mojo_time
        label = "faster" if ratio >= 1.0 else "slower"
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{upstream_time * 1e3:.3f} ms | {ratio:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
