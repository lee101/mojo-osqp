"""OSQP's Python interface backed by a Mojo ADMM solver."""

from .interface import (
    OSQP,
    OSQPException,
    SolverError,
    SolverStatus,
    algebra_available,
    algebras_available,
    constant,
    default_algebra,
)

__version__ = "0.1.0"

__all__ = [
    "OSQP",
    "OSQPException",
    "SolverError",
    "SolverStatus",
    "algebra_available",
    "algebras_available",
    "constant",
    "default_algebra",
]
