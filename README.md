# mojo-osqp

`mojo-osqp` is an open-source Mojo implementation of the operator-splitting
quadratic-programming solve path exposed through an OSQP-compatible Python API.
It solves convex problems of the form

```text
minimize    1/2 x' P x + q' x
subject to  l <= A x <= u
```

The implementation is useful for small and moderately sized dense QPs. It is
not a binding to upstream OSQP: factorization, ADMM iterations, projection,
residuals, adaptive `rho`, and stopping checks execute in Mojo.

```python
import numpy as np
from scipy import sparse
import mojo_osqp as osqp

P = sparse.csc_matrix([[4.0, 1.0], [1.0, 2.0]])
q = np.array([1.0, 1.0])
A = sparse.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
l = np.array([1.0, 0.0, 0.0])
u = np.array([1.0, 0.7, 0.7])

solver = osqp.OSQP()
solver.setup(
    P=P, q=q, A=A, l=l, u=u, verbose=False, eps_abs=1e-7, eps_rel=1e-7
)
result = solver.solve()
print(result.info.status, np.round(result.x, 6))
# solved [0.3 0.7]
```

## Coverage

The covered API mirrors upstream names and signatures:

- `OSQP.setup(P, q, A, l, u, **settings)` with SciPy sparse inputs, missing
  `P`, `q`, or constraints, infinite/one-sided bounds, and upper-triangular
  `P`
- `OSQP.solve(raise_error=None)` and the upstream-shaped result containing
  `x`, `y`, NaN certificate placeholders, and `info`
- `OSQP.update(q=..., l=..., u=..., Px=..., Px_idx=..., Ax=..., Ax_idx=...)`
- `OSQP.update_settings(...)` and `OSQP.warm_start(x=..., y=...)`
- `constant`, `SolverStatus`, `default_algebra`, `algebras_available`, and
  `algebra_available`

The numerical contract is tested against upstream `osqp` 1.1.3 on its
published two-variable example, unconstrained, equality-constrained, box,
one-sided, updated, warm-started, and randomized feasible problems.

This is deliberately not the whole upstream package. The current backend
densifies `P` and `A`, uses a dense Cholesky factor, and therefore needs
`O(n² + mn)` memory. It does not yet implement sparse QDLDL, scaling,
polishing, infeasibility certificates, indirect CG, derivatives, code
generation, CUDA algebras, or signal/time-limit interruption. `scaling`,
`polishing`, and the time-limit-related settings are accepted for setup-call
compatibility but currently have no effect. `codegen`, derivatives, and the
indirect solver raise `NotImplementedError` instead of silently changing
semantics. Only solved, maximum-iteration, and non-convex-factorization
statuses are currently produced.

These boundaries matter: upstream OSQP remains the right choice for very
large sparse problems or applications that require certificates. This
backend is aimed at dense control, allocation, and prototyping workloads
where a compact Mojo solver is useful.

## Install and run

The Pixi environment includes the pinned Mojo nightly, NumPy, SciPy, pytest,
and upstream OSQP:

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` compiles the single Mojo compilation unit to
`dist/libmojo-osqp.so`. With `PYTHONPATH=python`, importing `mojo_osqp` loads
that library. If it is missing or older than the source, the Python bridge
invokes `build/build.sh`. A prebuilt library can instead be selected with
`MOJO_OSQP_LIB=/absolute/path/libmojo-osqp.so`.

## Performance

Measured by the final `pixi run bench` pass on an Intel Xeon E5-2697 v4 at
2.30 GHz, Linux x86-64, using mojo-osqp 0.1.0, upstream OSQP 1.1.3, NumPy
2.5.1, and SciPy 1.18.0. Each entry is the best of 5 wall-clock runs per
implementation and case. Times include the Python API and memory conversions
described by each case. Lower is better; the relative column is upstream time
divided by Mojo time.

| case | mojo-osqp | upstream osqp | relative |
| --- | ---: | ---: | ---: |
| dense setup + solve (32 vars, 64 cons) | 0.904 ms | 1.909 ms | 2.11x faster |
| dense setup + solve (96 vars, 192 cons) | 4.552 ms | 7.260 ms | 1.60x faster |
| box setup + solve (256 vars) | 0.482 ms | 1.167 ms | 2.42x faster |
| warm solve (96 vars, 192 cons) | 0.499 ms | 1.359 ms | 2.72x faster |
| q update + warm solve (96 vars, 192 cons) | 0.524 ms | 1.270 ms | 2.42x faster |

The Mojo backend is faster in all five measured cases. Dense kernels use the
host's native `float64` SIMD width with scalar remainder loops. Matrix-vector
products stay serial at the supported problem sizes because their work is
memory-bound and thread-launch overhead does not repay itself. Diagonal `P`
and `A` use an O(n) factor and direct
elementwise solve instead of allocating or factoring dense n-by-n matrices.

There is no GPU path. The solver's hot operations at these sizes are
matrix-vector products, triangular solves, and unblocked rank updates/dot
products with insufficient arithmetic intensity to repay device transfer and
kernel-launch overhead.

## How it works

ADMM introduces `z = Ax`. With a per-constraint penalty `rho`, every iteration
solves

```text
(P + sigma I + A' diag(rho) A) x =
    sigma x_old - q + A' (rho z - y)
```

then performs over-relaxation, projection of `z` onto `[l, u]`, and the dual
update. Infinity-norm primal and dual residuals drive termination. Equality
rows receive the same larger penalty used by OSQP, and adaptive residual
balancing can rescale `rho` and refactor the system. The factor is retained
across warm solves and vector-only `q` updates.

Python owns every allocation. SciPy CSC inputs are canonicalized and converted
once to row-major contiguous `float64` arrays, or contiguous diagonal vectors
when both matrices are diagonal. `ctypes` passes their addresses as 64-bit
integers because exported Mojo functions cannot carry parametric pointer
origins across the C ABI. The exported functions use
`@export("name")` and `abi("C")`; inside Mojo the addresses are reconstructed
as `UnsafePointer[Float64, AnyOrigin[mut=True]]`. Every call also passes and
validates each buffer length, rejects null non-empty buffers, and checks
dimensions before reconstructing pointers. Python retains all owning NumPy
arrays for the complete synchronous call. ADMM iterations solve directly in
the persistent `x` buffer, and vector-only updates retain the matrix and factor
buffers. No Mojo allocation or Python callback occurs inside the solve loop.

## License

MIT
