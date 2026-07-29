"""An OSQP-compatible Python interface backed by Mojo."""

from __future__ import annotations

import time
import warnings
from enum import IntEnum
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from ._lib import addr, f64, lib

OSQP_INFTY = 1.0e30


class SolverStatus(IntEnum):
    OSQP_SOLVED = 1
    OSQP_SOLVED_INACCURATE = 2
    OSQP_PRIMAL_INFEASIBLE = 3
    OSQP_PRIMAL_INFEASIBLE_INACCURATE = 4
    OSQP_DUAL_INFEASIBLE = 5
    OSQP_DUAL_INFEASIBLE_INACCURATE = 6
    OSQP_MAX_ITER_REACHED = 7
    OSQP_TIME_LIMIT_REACHED = 8
    OSQP_NON_CVX = 9
    OSQP_SIGINT = 10
    OSQP_UNSOLVED = 11


_STATUS_TEXT = {
    1: "solved",
    2: "solved inaccurate",
    3: "primal infeasible",
    4: "primal infeasible inaccurate",
    5: "dual infeasible",
    6: "dual infeasible inaccurate",
    7: "maximum iterations reached",
    8: "run time limit reached",
    9: "problem non convex",
    10: "interrupted",
    11: "unsolved",
}


class OSQPException(Exception):
    pass


class SolverError(OSQPException):
    pass


_DEFAULTS = {
    "verbose": True,
    "warm_starting": True,
    "scaling": 10,
    "polishing": False,
    "rho": 0.1,
    "rho_is_vec": True,
    "sigma": 1.0e-6,
    "alpha": 1.6,
    "adaptive_rho": True,
    "adaptive_rho_interval": 50,
    "adaptive_rho_tolerance": 5.0,
    "max_iter": 4000,
    "eps_abs": 1.0e-3,
    "eps_rel": 1.0e-3,
    "eps_prim_inf": 1.0e-4,
    "eps_dual_inf": 1.0e-4,
    "scaled_termination": False,
    "check_termination": 25,
    "check_dualgap": True,
    "time_limit": 1.0e10,
    "delta": 1.0e-6,
    "polish_refine_iter": 3,
}

_ACCEPTED_UNUSED = {
    "device",
    "linsys_solver",
    "cg_max_iter",
    "cg_tol_reduction",
    "cg_tol_fraction",
    "cg_precond",
    "adaptive_rho_fraction",
}

_IGNORED_SETTINGS = {
    "verbose",
    "scaling",
    "polishing",
    "eps_prim_inf",
    "eps_dual_inf",
    "scaled_termination",
    "check_dualgap",
    "time_limit",
    "delta",
    "polish_refine_iter",
}


def _as_csc(matrix, shape, name):
    if matrix is None:
        return sparse.csc_matrix(shape, dtype=np.float64)
    if isinstance(matrix, np.ndarray) and matrix.ndim == 2:
        raise TypeError(f"{name} is required to be a sparse matrix")
    if not sparse.issparse(matrix):
        raise TypeError(f"{name} is required to be a sparse matrix")
    if not sparse.isspmatrix_csc(matrix):
        warnings.warn(f"Converting sparse {name} to a CSC matrix. This may take a while...")
        matrix = matrix.tocsc()
    else:
        matrix = matrix.copy()
    matrix.sort_indices()
    matrix = matrix.astype(np.float64)
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _has_diagonal_structure(matrix):
    columns = np.repeat(np.arange(matrix.shape[1]), np.diff(matrix.indptr))
    return np.array_equal(matrix.indices, columns)


class OSQP:
    """Dense direct ADMM solver with the covered upstream OSQP method names."""

    def __init__(self, *args, **kwargs):
        if args:
            raise TypeError("OSQP() accepts keyword arguments only")
        algebra = kwargs.pop("algebra", "builtin")
        if kwargs:
            raise TypeError(f"unexpected arguments: {list(kwargs)}")
        if algebra != "builtin":
            raise RuntimeError(f"Algebra {algebra} not available")
        self.algebra = algebra
        self.n = None
        self.m = None
        self.settings = SimpleNamespace(**_DEFAULTS)
        self._ready = False
        self._setup_time = 0.0
        self._update_time = 0.0

    def setup(self, P, q, A, l, u, **settings):
        started = time.perf_counter()
        if P is None:
            if q is not None:
                n = len(q)
            elif A is not None:
                n = A.shape[1]
            else:
                raise ValueError("The problem does not have any variables")
        else:
            n = P.shape[0]
            if P.shape != (n, n):
                raise ValueError("P must be square")
        m = 0 if A is None else A.shape[0]
        if A is not None and A.shape[1] != n:
            raise ValueError("Incorrect dimension of A")
        if A is None:
            assert l is None and u is None, "If A is unspecified, leave l/u unspecified too."
        else:
            assert l is not None or u is not None, "If A is specified, specify at least one of l/u."

        P_csc = _as_csc(P, (n, n), "P")
        if sparse.tril(P_csc, -1).nnz:
            P_csc = sparse.triu(P_csc, format="csc")
        A_csc = _as_csc(A, (m, n), "A")
        q_array = np.zeros(n) if q is None else f64(q).ravel()
        lower = np.zeros(m) if A is None else (
            np.full(m, -OSQP_INFTY) if l is None else f64(l).ravel()
        )
        upper = np.zeros(m) if A is None else (
            np.full(m, OSQP_INFTY) if u is None else f64(u).ravel()
        )
        if q_array.size != n:
            raise ValueError("Incorrect dimension of q")
        if lower.size != m:
            raise ValueError("Incorrect dimension of l")
        if upper.size != m:
            raise ValueError("Incorrect dimension of u")
        if not np.all(np.isfinite(q_array)):
            raise ValueError("q must contain only finite values")
        lower = np.maximum(lower, -OSQP_INFTY)
        upper = np.minimum(upper, OSQP_INFTY)
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            raise ValueError("bounds must not contain NaN")
        if np.any(lower > upper):
            raise OSQPException("Lower bound at index is greater than upper bound")

        self.n, self.m = n, m
        self._P_csc = P_csc
        self._A_csc = A_csc
        self._diagonal = (
            m == n
            and _has_diagonal_structure(P_csc)
            and _has_diagonal_structure(A_csc)
        )
        self._q = np.ascontiguousarray(q_array)
        self._l = np.ascontiguousarray(lower)
        self._u = np.ascontiguousarray(upper)
        self._make_dense_matrices()
        self.settings = SimpleNamespace(**_DEFAULTS)
        self.update_settings(**settings)
        self._allocate_state()
        self._refactor()
        self._ready = True
        self._setup_time = time.perf_counter() - started
        return None

    def _make_dense_matrices(self):
        if self._diagonal:
            self._P = np.ascontiguousarray(self._P_csc.diagonal())
            self._A = np.ascontiguousarray(self._A_csc.diagonal())
            return
        upper = self._P_csc.toarray()
        self._P = np.ascontiguousarray(upper + np.triu(upper, 1).T)
        self._A = np.ascontiguousarray(self._A_csc.toarray())

    def _allocate_state(self):
        n, m = self.n, self.m
        factor_shape = n if self._diagonal else (n, n)
        self._factor = np.empty(factor_shape, dtype=np.float64)
        self._x = np.zeros(n, dtype=np.float64)
        self._y = np.zeros(m, dtype=np.float64)
        self._z = np.minimum(self._u, np.maximum(self._l, np.zeros(m)))
        self._ax = np.empty(m, dtype=np.float64)
        self._px = np.empty(n, dtype=np.float64)
        self._aty = np.empty(n, dtype=np.float64)
        self._rhs = np.empty(n, dtype=np.float64)
        self._info_buffer = np.empty(6, dtype=np.float64)
        self._set_rho_vector()

    def _set_rho_vector(self):
        base = float(self.settings.rho)
        self._rho = np.full(self.m, base, dtype=np.float64)
        equality = np.isfinite(self._l) & np.isfinite(self._u) & (
            np.abs(self._u - self._l) <= 1.0e-12
        )
        self._rho[equality] *= 1000.0

    def _refactor(self):
        ok = lib().mosqp_factor(
            addr(self._P),
            addr(self._A),
            addr(self._rho),
            addr(self._factor),
            self.n,
            self.m,
            int(self._diagonal),
            self._P.size,
            self._A.size,
            self._rho.size,
            self._factor.size,
            float(self.settings.sigma),
        )
        if ok < 0:
            raise SolverError("native factorization rejected invalid buffers or settings")
        if not ok:
            raise OSQPException(SolverStatus.OSQP_NON_CVX)

    def solve(self, raise_error=None):
        if not self._ready:
            raise RuntimeError("setup must be called before solve")
        if raise_error is None:
            warnings.warn(
                "The default value of raise_error will change to True in the future.",
                PendingDeprecationWarning,
            )
            raise_error = False
        if not self.settings.warm_starting:
            self._x.fill(0.0)
            self._y.fill(0.0)
            self._z[:] = np.minimum(self._u, np.maximum(self._l, 0.0))

        started = time.perf_counter()
        adaptive_interval = (
            int(self.settings.adaptive_rho_interval)
            if self.settings.adaptive_rho
            else 0
        )
        status = lib().mosqp_solve(
            addr(self._P),
            addr(self._A),
            addr(self._q),
            addr(self._l),
            addr(self._u),
            addr(self._rho),
            addr(self._factor),
            addr(self._x),
            addr(self._y),
            addr(self._z),
            addr(self._ax),
            addr(self._px),
            addr(self._aty),
            addr(self._rhs),
            addr(self._info_buffer),
            self.n,
            self.m,
            int(self.settings.max_iter),
            int(self.settings.check_termination),
            adaptive_interval,
            int(self._diagonal),
            self._P.size,
            self._A.size,
            self._q.size,
            self._l.size,
            self._u.size,
            self._rho.size,
            self._factor.size,
            self._x.size,
            self._y.size,
            self._z.size,
            self._ax.size,
            self._px.size,
            self._aty.size,
            self._rhs.size,
            self._info_buffer.size,
            float(self.settings.eps_abs),
            float(self.settings.eps_rel),
            float(self.settings.alpha),
            float(self.settings.sigma),
            float(self.settings.adaptive_rho_tolerance),
        )
        solve_time = time.perf_counter() - started
        if status < 0:
            raise SolverError("native solver rejected invalid buffers or settings")
        values = self._info_buffer
        info = SimpleNamespace(
            _pybind11_conduit_v1_=None,
            status=_STATUS_TEXT[status],
            status_val=status,
            status_polish=0,
            obj_val=float(values[1]),
            prim_res=float(values[2]),
            dual_res=float(values[3]),
            iter=int(values[0]),
            rho_updates=int(values[4]),
            rho_estimate=float(values[5]),
            setup_time=self._setup_time,
            solve_time=solve_time,
            update_time=self._update_time,
            polish_time=0.0,
            run_time=self._setup_time + self._update_time + solve_time,
            primdual_int=np.nan,
            duality_gap=np.nan,
            rel_kkt_error=np.nan,
        )
        if status != SolverStatus.OSQP_SOLVED and raise_error:
            raise OSQPException(status)
        return SimpleNamespace(
            x=self._x.copy(),
            y=self._y.copy(),
            prim_inf_cert=np.full(self.m, np.nan),
            dual_inf_cert=np.full(self.n, np.nan),
            info=info,
        )

    def update(self, **kwargs):
        if not self._ready:
            raise RuntimeError("setup must be called before update")
        started = time.perf_counter()
        q, lower, upper = kwargs.pop("q", None), kwargs.pop("l", None), kwargs.pop("u", None)
        bounds_changed = lower is not None or upper is not None
        next_q = self._q.copy()
        next_l = self._l.copy()
        next_u = self._u.copy()
        if q is not None:
            value = f64(q).ravel()
            if value.size != self.n:
                raise ValueError("Incorrect dimension of q")
            if not np.all(np.isfinite(value)):
                raise ValueError("q must contain only finite values")
            next_q[:] = value
        if lower is not None:
            value = np.maximum(f64(lower).ravel(), -OSQP_INFTY)
            if value.size != self.m:
                raise ValueError("Incorrect dimension of l")
            next_l[:] = value
        if upper is not None:
            value = np.minimum(f64(upper).ravel(), OSQP_INFTY)
            if value.size != self.m:
                raise ValueError("Incorrect dimension of u")
            next_u[:] = value
        if np.any(np.isnan(next_l)) or np.any(np.isnan(next_u)):
            raise ValueError("bounds must not contain NaN")
        if np.any(next_l > next_u):
            raise OSQPException("Lower bound at index is greater than upper bound")

        matrix_changed = False
        unknown = set(kwargs) - {"Px", "Px_idx", "Ax", "Ax_idx"}
        if unknown:
            raise ValueError(f"Unrecognized update fields {sorted(unknown)}")
        next_matrices = {"P": self._P_csc.copy(), "A": self._A_csc.copy()}
        for name, matrix in next_matrices.items():
            values = kwargs.pop(f"{name}x", None)
            indices = kwargs.pop(f"{name}x_idx", None)
            if values is None and indices is not None:
                raise ValueError(f"{name}x_idx requires {name}x")
            if values is not None:
                values = f64(values).ravel()
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"{name}x must contain only finite values")
                if indices is None:
                    if len(values) != len(matrix.data):
                        raise ValueError(
                            f"new number of elements ({len(values)}) out of bounds for {name}"
                        )
                    matrix.data[:] = values
                else:
                    raw_indices = np.asarray(indices)
                    if not np.issubdtype(raw_indices.dtype, np.integer):
                        raise TypeError(f"{name}x_idx must contain integers")
                    indices = np.asarray(raw_indices, dtype=np.intp).ravel()
                    if indices.size != values.size:
                        raise ValueError(
                            f"{name}x and {name}x_idx must have the same length"
                        )
                    if np.any(indices < 0) or np.any(indices >= matrix.data.size):
                        raise IndexError(f"{name}x_idx contains an out-of-range index")
                    matrix.data[indices] = values
                matrix_changed = True
        self._q[:] = next_q
        self._l[:] = next_l
        self._u[:] = next_u
        if matrix_changed:
            self._P_csc = next_matrices["P"]
            self._A_csc = next_matrices["A"]
            self._make_dense_matrices()
            self._refactor()
        elif bounds_changed:
            self._set_rho_vector()
            self._refactor()
        self._update_time = time.perf_counter() - started
        return None

    def update_settings(self, **kwargs):
        renamed = {"polish": "polishing", "warm_start": "warm_starting"}
        for old, new in renamed.items():
            if old in kwargs:
                warnings.warn(
                    f'"{old}" is deprecated. Please use "{new}" instead.',
                    DeprecationWarning,
                )
                kwargs[new] = kwargs.pop(old)
        if "solver_type" in kwargs:
            solver_type = kwargs.pop("solver_type")
            if solver_type != "direct":
                raise NotImplementedError("the Mojo backend implements the direct solver")
        if "cg_preconditioner" in kwargs:
            if kwargs.pop("cg_preconditioner") not in (None, "diagonal"):
                raise ValueError("cg_preconditioner must be None or 'diagonal'")
        refactor = False
        validators = {
            "rho": lambda value: np.isfinite(value) and value > 0,
            "sigma": lambda value: np.isfinite(value) and value > 0,
            "alpha": lambda value: np.isfinite(value) and 0 < value < 2,
            "max_iter": lambda value: isinstance(value, (int, np.integer)) and value > 0,
            "check_termination": lambda value: isinstance(value, (int, np.integer)) and value >= 0,
            "adaptive_rho_interval": lambda value: isinstance(value, (int, np.integer)) and value >= 0,
            "adaptive_rho_tolerance": lambda value: np.isfinite(value) and value >= 1,
            "eps_abs": lambda value: np.isfinite(value) and value >= 0,
            "eps_rel": lambda value: np.isfinite(value) and value >= 0,
        }
        for key, value in list(kwargs.items()):
            if key in _ACCEPTED_UNUSED:
                raise NotImplementedError(
                    f"setting {key} is not implemented by the Mojo backend"
                )
            elif key in _DEFAULTS:
                if key in validators and not validators[key](value):
                    raise ValueError(f"invalid value for setting {key}: {value!r}")
                if (
                    key in _IGNORED_SETTINGS
                    and value != _DEFAULTS[key]
                    and key not in {"verbose", "polishing"}
                ):
                    warnings.warn(
                        f"setting {key} is accepted for compatibility but has no effect",
                        RuntimeWarning,
                    )
                if key == "polishing" and value:
                    warnings.warn(
                        "polishing is accepted for compatibility but is not implemented",
                        RuntimeWarning,
                    )
                setattr(self.settings, key, kwargs.pop(key))
                refactor |= key in {"rho", "sigma"}
        if kwargs:
            raise ValueError(f"Unrecognized settings {list(kwargs)}")
        if self._ready and refactor:
            self._set_rho_vector()
            self._refactor()
        return None

    def warm_start(self, x=None, y=None):
        if x is not None:
            value = f64(x).ravel()
            if value.size != self.n:
                raise ValueError("Incorrect dimension of x")
            if not np.all(np.isfinite(value)):
                raise ValueError("x must contain only finite values")
            self._x[:] = value
        if y is not None:
            value = f64(y).ravel()
            if value.size != self.m:
                raise ValueError("Incorrect dimension of y")
            if not np.all(np.isfinite(value)):
                raise ValueError("y must contain only finite values")
            self._y[:] = value
        if self._diagonal:
            np.multiply(self._A, self._x, out=self._ax)
        else:
            self._ax[:] = self._A @ self._x
        self._z[:] = np.minimum(
            self._u, np.maximum(self._l, self._ax + self._y / self._rho)
        )
        return None

    def constant(self, which):
        return constant(which)

    def codegen(self, *args, **kwargs):
        raise NotImplementedError("code generation is not covered by mojo-osqp")

    def adjoint_derivative_compute(self, *args, **kwargs):
        raise NotImplementedError("adjoint derivatives are not covered by mojo-osqp")


_CONSTANTS = {
    "OSQP_INFTY": OSQP_INFTY,
    "OSQP_NAN": np.nan,
    **{status.name: int(status) for status in SolverStatus},
}


def constant(which, algebra="builtin"):
    if algebra != "builtin":
        raise RuntimeError(f"Algebra {algebra} not available")
    try:
        return _CONSTANTS[which]
    except KeyError as error:
        raise RuntimeError(f"Unknown constant {which}") from error


def default_algebra():
    return "builtin"


def algebras_available():
    return ["builtin"]


def algebra_available(algebra):
    return algebra == "builtin"
