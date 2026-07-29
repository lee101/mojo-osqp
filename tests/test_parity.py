"""Numerical and behavioural parity with upstream osqp."""

from __future__ import annotations

import warnings

import numpy as np
import osqp
import pytest
from scipy import sparse

import mojo_osqp
from mojo_osqp._lib import addr, f64, lib


TIGHT = dict(verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=10_000)


def solve_pair(P, q, A=None, l=None, u=None, **settings):
    options = TIGHT | settings
    ours = mojo_osqp.OSQP()
    theirs = osqp.OSQP()
    ours.setup(P=P, q=q, A=A, l=l, u=u, **options)
    theirs.setup(P=P, q=q, A=A, l=l, u=u, **options)
    return ours, theirs, ours.solve(), theirs.solve()


def assert_solutions_match(ours, theirs, atol=2e-5):
    assert ours.info.status_val == osqp.SolverStatus.OSQP_SOLVED
    assert theirs.info.status_val == osqp.SolverStatus.OSQP_SOLVED
    assert np.allclose(ours.x, theirs.x, atol=atol, rtol=atol)
    assert ours.info.obj_val == pytest.approx(theirs.info.obj_val, abs=atol, rel=atol)
    assert ours.info.prim_res < 2e-5
    assert ours.info.dual_res < 2e-5


def official_problem():
    P = sparse.csc_matrix([[4.0, 1.0], [1.0, 2.0]])
    q = np.array([1.0, 1.0])
    A = sparse.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    l = np.array([1.0, 0.0, 0.0])
    u = np.array([1.0, 0.7, 0.7])
    return P, q, A, l, u


def test_official_osqp_example():
    _, _, ours, theirs = solve_pair(*official_problem())
    assert_solutions_match(ours, theirs, atol=2e-7)
    assert np.allclose(ours.x, [0.3, 0.7], atol=2e-7)
    assert np.allclose(ours.y, theirs.y, atol=2e-6)


def test_unconstrained_quadratic():
    P = sparse.csc_matrix([[5.0, 1.0], [1.0, 3.0]])
    q = np.array([-2.0, 4.0])
    _, _, ours, theirs = solve_pair(P, q)
    assert_solutions_match(ours, theirs, atol=1e-7)
    assert np.allclose(ours.x, -np.linalg.solve(P.toarray(), q), atol=1e-7)


def test_missing_p_and_q():
    A = sparse.eye(2, format="csc")
    l = np.array([-1.0, -1.0])
    u = np.array([1.0, 1.0])
    _, _, ours, theirs = solve_pair(None, None, A, l, u)
    assert_solutions_match(ours, theirs, atol=1e-7)


def test_equality_constrained_problem():
    P = sparse.eye(3, format="csc")
    q = np.array([-1.0, -2.0, -3.0])
    A = sparse.csc_matrix([[1.0, 1.0, 1.0]])
    bounds = np.array([1.0])
    _, _, ours, theirs = solve_pair(P, q, A, bounds, bounds)
    assert_solutions_match(ours, theirs, atol=2e-7)
    assert np.sum(ours.x) == pytest.approx(1.0, abs=2e-7)


def test_box_constraints():
    n = 12
    P = sparse.diags(np.linspace(1.0, 3.0, n), format="csc")
    q = np.linspace(-2.0, 2.0, n)
    A = sparse.eye(n, format="csc")
    l = np.full(n, -0.4)
    u = np.full(n, 0.7)
    _, _, ours, theirs = solve_pair(P, q, A, l, u)
    assert_solutions_match(ours, theirs, atol=3e-7)


def test_diagonal_path_with_simd_tail():
    n = 13
    P = sparse.diags(np.linspace(0.7, 2.3, n), format="csc")
    A = sparse.diags(np.linspace(0.8, 1.2, n), format="csc")
    q = np.linspace(-1.5, 1.5, n)
    l = np.full(n, -0.35)
    u = np.full(n, 0.55)
    ours, theirs, ours_result, theirs_result = solve_pair(P, q, A, l, u)
    assert ours._diagonal
    assert ours._P.shape == (n,)
    assert ours._A.shape == (n,)
    assert ours._factor.shape == (n,)
    assert_solutions_match(ours_result, theirs_result, atol=3e-7)
    ours.warm_start(x=ours_result.x, y=ours_result.y)
    warm_result = ours.solve()
    assert np.allclose(warm_result.x, ours_result.x, atol=3e-7, rtol=3e-7)
    new_px = np.linspace(0.9, 2.5, n)
    ours.update(Px=new_px)
    theirs.update(Px=new_px)
    assert_solutions_match(ours.solve(), theirs.solve(), atol=3e-7)


def test_matvec_simd_tail_and_parallel_threshold():
    rng = np.random.default_rng(91)
    for rows, cols in ((5, 13), (257, 511)):
        matrix = np.ascontiguousarray(rng.normal(size=(rows, cols)))
        vector = np.ascontiguousarray(rng.normal(size=cols))
        result = np.empty(rows, dtype=np.float64)
        status = lib().mosqp_matvec(
            addr(matrix), addr(vector), addr(result), rows, cols,
            matrix.size, vector.size, result.size,
        )
        assert status == 0
        assert np.allclose(result, matrix @ vector, atol=2e-12, rtol=2e-12)


def test_native_boundary_rejects_bad_lengths_and_python_rejects_layout():
    matrix = np.eye(3, dtype=np.float64)
    vector = np.ones(3, dtype=np.float64)
    result = np.empty(3, dtype=np.float64)
    assert lib().mosqp_matvec(
        addr(matrix), addr(vector), addr(result), 3, 3,
        matrix.size - 1, vector.size, result.size,
    ) == -1
    with pytest.raises(TypeError, match="float64"):
        addr(np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        addr(np.ones((3, 2), dtype=np.float64)[:, 0])
    read_only = np.ones(3, dtype=np.float64)
    read_only.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        addr(read_only)
    with pytest.raises(TypeError, match="complex"):
        f64(np.array([1 + 2j]))
    with pytest.raises(ValueError, match="exact float64"):
        f64(np.array([2**53 + 1], dtype=np.int64))


@pytest.mark.parametrize("seed", range(5))
def test_random_feasible_dense_qp(seed):
    rng = np.random.default_rng(seed)
    n, m = 8, 14
    R = rng.normal(size=(n, n))
    P_dense = R.T @ R + 0.5 * np.eye(n)
    A_dense = rng.normal(size=(m, n))
    feasible = rng.normal(size=n)
    center = A_dense @ feasible
    width = rng.uniform(0.2, 1.0, size=m)
    P = sparse.csc_matrix(P_dense)
    A = sparse.csc_matrix(A_dense)
    _, _, ours, theirs = solve_pair(
        P, rng.normal(size=n), A, center - width, center + width
    )
    assert_solutions_match(ours, theirs, atol=8e-5)


def test_one_sided_and_infinite_bounds():
    P = sparse.eye(3, format="csc")
    q = np.array([-2.0, 1.0, -0.5])
    A = sparse.eye(3, format="csc")
    l = np.array([0.0, -np.inf, -1.0])
    u = np.array([np.inf, 0.2, np.inf])
    _, _, ours, theirs = solve_pair(P, q, A, l, u)
    assert_solutions_match(ours, theirs, atol=2e-6)


def test_upper_triangular_P_matches_upstream():
    P = sparse.csc_matrix([[4.0, 1.5], [0.0, 2.0]])
    q = np.array([-1.0, -1.0])
    _, _, ours, theirs = solve_pair(P, q)
    assert_solutions_match(ours, theirs, atol=1e-7)


def test_update_vectors_matches_upstream():
    P, q, A, l, u = official_problem()
    ours, theirs, _, _ = solve_pair(P, q, A, l, u)
    new_q = np.array([-0.5, 0.25])
    new_l = np.array([0.8, -0.1, 0.0])
    new_u = np.array([0.8, 0.9, 0.8])
    ours.update(q=new_q, l=new_l, u=new_u)
    theirs.update(q=new_q, l=new_l, u=new_u)
    assert_solutions_match(ours.solve(), theirs.solve(), atol=3e-6)


def test_update_all_matrix_values():
    P, q, A, l, u = official_problem()
    ours, theirs, _, _ = solve_pair(P, q, A, l, u)
    new_px = np.array([5.0, 0.5, 3.0])
    new_ax = np.array([1.2, 1.0, 0.8, 1.1])
    ours.update(Px=new_px, Ax=new_ax)
    theirs.update(Px=new_px, Ax=new_ax)
    assert_solutions_match(ours.solve(), theirs.solve(), atol=4e-6)


def test_indexed_matrix_update():
    P, q, A, l, u = official_problem()
    ours, theirs, _, _ = solve_pair(P, q, A, l, u)
    ours.update(Px=np.array([6.0]), Px_idx=np.array([0]))
    theirs.update(Px=np.array([6.0]), Px_idx=np.array([0]))
    assert_solutions_match(ours.solve(), theirs.solve(), atol=3e-6)


def test_indexed_a_matrix_update():
    P, q, A, l, u = official_problem()
    ours, theirs, _, _ = solve_pair(P, q, A, l, u)
    ours.update(Ax=np.array([1.2]), Ax_idx=np.array([0]))
    theirs.update(Ax=np.array([1.2]), Ax_idx=np.array([0]))
    assert_solutions_match(ours.solve(), theirs.solve(), atol=3e-6)


def test_warm_start_solution_and_iteration_count():
    P, q, A, l, u = official_problem()
    solver = mojo_osqp.OSQP()
    solver.setup(P=P, q=q, A=A, l=l, u=u, **TIGHT)
    cold = solver.solve()
    solver.warm_start(x=cold.x, y=cold.y)
    warm = solver.solve()
    assert np.allclose(warm.x, cold.x, atol=2e-7)
    assert warm.info.iter <= cold.info.iter


def test_update_settings_rho_and_tolerances():
    P, q, A, l, u = official_problem()
    solver = mojo_osqp.OSQP()
    solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
    solver.update_settings(rho=0.4, eps_abs=1e-8, eps_rel=1e-8, max_iter=8000)
    result = solver.solve()
    assert result.info.status == "solved"
    assert np.allclose(result.x, [0.3, 0.7], atol=2e-7)


def test_result_contract():
    solver = mojo_osqp.OSQP()
    P, q, A, l, u = official_problem()
    solver.setup(P=P, q=q, A=A, l=l, u=u, **TIGHT)
    result = solver.solve()
    assert result.x.shape == (2,)
    assert result.y.shape == (3,)
    assert result.prim_inf_cert.shape == (3,)
    assert result.dual_inf_cert.shape == (2,)
    for name in (
        "status",
        "status_val",
        "obj_val",
        "iter",
        "prim_res",
        "dual_res",
        "setup_time",
        "solve_time",
        "run_time",
        "rho_updates",
    ):
        assert hasattr(result.info, name)


def test_max_iter_status_and_raise_error():
    P, q, A, l, u = official_problem()
    solver = mojo_osqp.OSQP()
    solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, max_iter=1)
    assert solver.solve().info.status_val == osqp.SolverStatus.OSQP_MAX_ITER_REACHED
    with pytest.raises(mojo_osqp.OSQPException):
        solver.solve(raise_error=True)


def test_constants_and_algebra_helpers():
    assert mojo_osqp.constant("OSQP_SOLVED") == osqp.constant("OSQP_SOLVED")
    assert mojo_osqp.constant("OSQP_INFTY") == osqp.constant("OSQP_INFTY")
    assert np.isnan(mojo_osqp.constant("OSQP_NAN"))
    assert mojo_osqp.default_algebra() == "builtin"
    assert mojo_osqp.algebras_available() == ["builtin"]
    assert mojo_osqp.algebra_available("builtin")
    assert not mojo_osqp.algebra_available("cuda")


def test_csr_conversion_and_dense_rejection():
    P = sparse.eye(2, format="csr")
    q = np.ones(2)
    solver = mojo_osqp.OSQP()
    with pytest.warns(UserWarning, match="Converting sparse P"):
        solver.setup(P=P, q=q, A=None, l=None, u=None, verbose=False)
    with pytest.raises(TypeError, match="sparse matrix"):
        mojo_osqp.OSQP().setup(
            P=np.eye(2), q=q, A=None, l=None, u=None, verbose=False
        )


def test_invalid_bounds_rejected():
    with pytest.raises(mojo_osqp.OSQPException):
        mojo_osqp.OSQP().setup(
            P=sparse.eye(1, format="csc"),
            q=np.zeros(1),
            A=sparse.eye(1, format="csc"),
            l=np.array([2.0]),
            u=np.array([1.0]),
            verbose=False,
        )


def test_invalid_dimensions_indices_values_and_settings_are_rejected():
    P, q, A, l, u = official_problem()
    solver = mojo_osqp.OSQP()
    with pytest.raises(ValueError, match="dimension of q"):
        solver.setup(P=P, q=np.ones(3), A=A, l=l, u=u, verbose=False)
    with pytest.raises(ValueError, match="finite"):
        solver.setup(P=P, q=np.array([np.nan, 0.0]), A=A, l=l, u=u, verbose=False)
    solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
    with pytest.raises(TypeError, match="integers"):
        solver.update(Px=[1.0], Px_idx=[0.5])
    with pytest.raises(IndexError, match="out-of-range"):
        solver.update(Px=[1.0], Px_idx=[-1])
    with pytest.raises(ValueError, match="same length"):
        solver.update(Px=[1.0, 2.0], Px_idx=[0])
    with pytest.raises(ValueError, match="setting sigma"):
        solver.update_settings(sigma=0.0)
    with pytest.raises(ValueError, match="setting max_iter"):
        solver.update_settings(max_iter=1.5)


def test_deprecated_setting_aliases():
    solver = mojo_osqp.OSQP()
    P, q, A, l, u = official_problem()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, polish=True)
    assert solver.settings.polishing is True
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert any(issubclass(item.category, RuntimeWarning) for item in caught)


def test_unsupported_features_are_explicit():
    solver = mojo_osqp.OSQP()
    with pytest.raises(NotImplementedError):
        solver.codegen("generated")
    with pytest.raises(NotImplementedError):
        solver.adjoint_derivative_compute()
    with pytest.raises(NotImplementedError):
        solver.update_settings(solver_type="indirect")
    with pytest.raises(NotImplementedError, match="device"):
        solver.update_settings(device="cuda")
