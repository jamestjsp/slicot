"""
Tests for MB02PD - Linear System Solver with LU Factorization.

Solves op(A)*X = B using LU factorization with optional equilibration
and iterative refinement for improved accuracy.

op(A) = A or A' (transpose).
"""

import numpy as np
import pytest


def test_mb02pd_basic():
    """
    Test MB02PD basic linear system solve.

    Solve A*X = B for a well-conditioned system.

    Random seed: 42 (for reproducibility)
    """
    from slicot import mb02pd

    np.random.seed(42)
    n = 4
    nrhs = 2

    A = np.eye(n) + 0.5 * np.random.randn(n, n)
    A = np.asfortranarray(A)

    X_true = np.random.randn(n, nrhs)
    B = A @ X_true
    B = np.asfortranarray(B)

    X, ferr, berr, rcond, info = mb02pd('N', 'N', A.copy(order='F'), B.copy(order='F'))

    assert info == 0, f"mb02pd failed with info={info}"
    assert X.shape == (n, nrhs)
    assert rcond > 0

    np.testing.assert_allclose(X, X_true, rtol=1e-12)


def test_mb02pd_transpose():
    """
    Test MB02PD with transpose: A'*X = B.

    Random seed: 123 (for reproducibility)
    """
    from slicot import mb02pd

    np.random.seed(123)
    n = 4
    nrhs = 1

    A = np.eye(n) + 0.3 * np.random.randn(n, n)
    A = np.asfortranarray(A)

    X_true = np.random.randn(n, nrhs)
    B = A.T @ X_true
    B = np.asfortranarray(B)

    X, ferr, berr, rcond, info = mb02pd('N', 'T', A.copy(order='F'), B.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(X, X_true, rtol=1e-12)


def test_mb02pd_identity():
    """
    Test MB02PD with identity matrix (trivial case).

    A = I, so X = B.
    """
    from slicot import mb02pd

    n = 3
    nrhs = 2

    A = np.eye(n, dtype=float, order='F')
    B = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]], order='F', dtype=float)

    X, ferr, berr, rcond, info = mb02pd('N', 'N', A.copy(order='F'), B.copy(order='F'))

    assert info == 0
    assert rcond == pytest.approx(1.0, rel=1e-10)
    np.testing.assert_allclose(X, B, rtol=1e-14)


def test_mb02pd_single_rhs():
    """
    Test MB02PD with single right-hand side.

    Random seed: 456 (for reproducibility)
    """
    from slicot import mb02pd

    np.random.seed(456)
    n = 5
    nrhs = 1

    A = 2.0 * np.eye(n) + 0.2 * np.random.randn(n, n)
    A = np.asfortranarray(A)

    x_true = np.random.randn(n, 1)
    b = A @ x_true
    b = np.asfortranarray(b)

    X, ferr, berr, rcond, info = mb02pd('N', 'N', A.copy(order='F'), b.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(X, x_true, rtol=1e-12)


def test_mb02pd_residual_check():
    """
    Validate residual: ||A*X - B|| should be small.

    Random seed: 789 (for reproducibility)
    """
    from slicot import mb02pd

    np.random.seed(789)
    n = 6
    nrhs = 3

    A = np.eye(n) + 0.4 * np.random.randn(n, n)
    A = np.asfortranarray(A)

    B = np.random.randn(n, nrhs)
    B = np.asfortranarray(B)

    A_copy = A.copy(order='F')
    B_copy = B.copy(order='F')

    X, ferr, berr, rcond, info = mb02pd('N', 'N', A_copy, B_copy)

    assert info == 0

    residual = A @ X - B
    rel_residual = np.linalg.norm(residual, 'fro') / np.linalg.norm(B, 'fro')

    assert rel_residual < 1e-12


def test_mb02pd_ill_conditioned():
    """
    Test MB02PD with moderately ill-conditioned matrix.

    Random seed: 111 (for reproducibility)
    """
    from slicot import mb02pd

    np.random.seed(111)
    n = 4

    U, _ = np.linalg.qr(np.random.randn(n, n))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    s = np.array([1.0, 1e-3, 1e-6, 1e-9])
    A = U @ np.diag(s) @ V.T
    A = np.asfortranarray(A)

    x_true = np.ones((n, 1))
    B = A @ x_true
    B = np.asfortranarray(B)

    X, ferr, berr, rcond, info = mb02pd('N', 'N', A.copy(order='F'), B.copy(order='F'))

    assert rcond < 1e-6, "rcond should be small for ill-conditioned matrix"


def test_mb02pd_n_equals_1():
    """
    Test MB02PD edge case with n=1 (scalar equation).
    """
    from slicot import mb02pd

    A = np.array([[2.0]], order='F', dtype=float)
    B = np.array([[6.0]], order='F', dtype=float)

    X, ferr, berr, rcond, info = mb02pd('N', 'N', A.copy(order='F'), B.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(X, [[3.0]], rtol=1e-14)


def test_mb02pd_multiple_rhs():
    """
    Test MB02PD with multiple right-hand sides.

    Random seed: 222 (for reproducibility)
    """
    from slicot import mb02pd

    np.random.seed(222)
    n = 4
    nrhs = 5

    A = np.eye(n) + 0.3 * np.random.randn(n, n)
    A = np.asfortranarray(A)

    X_true = np.random.randn(n, nrhs)
    B = A @ X_true
    B = np.asfortranarray(B)

    X, ferr, berr, rcond, info = mb02pd('N', 'N', A.copy(order='F'), B.copy(order='F'))

    assert info == 0
    assert X.shape == (n, nrhs)
    assert ferr.shape == (nrhs,)
    assert berr.shape == (nrhs,)

    np.testing.assert_allclose(X, X_true, rtol=1e-11)


def test_mb02pd_diagonal_matrix():
    """
    Test MB02PD with diagonal matrix.
    """
    from slicot import mb02pd

    n = 4
    nrhs = 2

    A = np.diag([2.0, 3.0, 4.0, 5.0]).astype(float, order='F')
    B = np.array([[2.0, 10.0],
                  [9.0, 15.0],
                  [12.0, 20.0],
                  [15.0, 25.0]], order='F', dtype=float)

    X_expected = np.array([[1.0, 5.0],
                           [3.0, 5.0],
                           [3.0, 5.0],
                           [3.0, 5.0]], order='F', dtype=float)

    X, ferr, berr, rcond, info = mb02pd('N', 'N', A.copy(order='F'), B.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(X, X_expected, rtol=1e-14)
