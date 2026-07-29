"""ctypes bridge to the Mojo ADMM kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_OSQP_LIB") or os.path.join(
    ROOT, "dist", "libmojo-osqp.so"
)
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mosqp_factor": ([I] * 11 + [F], I),
    "mosqp_matvec": ([I] * 8, I),
    "mosqp_solve": ([I] * 36 + [F] * 5, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "osqp.mojo")
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(source)
    ):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(value, *, copy: bool = False) -> np.ndarray:
    original = np.asarray(value)
    if np.issubdtype(original.dtype, np.complexfloating):
        raise TypeError("complex values are not supported")
    if np.issubdtype(original.dtype, np.floating) and original.dtype.itemsize > 8:
        raise TypeError("floating-point values wider than float64 are not supported")
    if np.issubdtype(original.dtype, np.integer) and original.size:
        maximum_exact_integer = 1 << 53
        if np.any(original > maximum_exact_integer) or np.any(
            original < -maximum_exact_integer
        ):
            raise ValueError("integer values outside the exact float64 range are not supported")
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def addr(value: np.ndarray) -> int:
    if not isinstance(value, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    if value.dtype != np.dtype(np.float64):
        raise TypeError("native buffers must have dtype float64")
    if not value.flags.c_contiguous:
        raise ValueError("native buffers must be C-contiguous")
    if not value.flags.writeable:
        raise ValueError("native buffers must be writeable")
    address = int(value.ctypes.data)
    if value.size and address == 0:
        raise ValueError("native buffer has a null data pointer")
    return address
