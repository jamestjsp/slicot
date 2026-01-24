"""
Tests for MB03XP: Periodic Schur decomposition of A*B product.

Computes Q' * A * Z = S and Z' * B * Q = T where:
- A is upper Hessenberg
- B is upper triangular
- S is real Schur form
- T is upper triangular

Uses numpy only - no scipy.
"""

import numpy as np


def test_mb03xp_n4_basic():
    """
    Test MB03XP with N=4 basic case.

    Random seed: 42 (for reproducibility)
    """
    from slicot import mb03xp

    np.random.seed(42)
    n = 4
    ilo = 1
    ihi = 4

    A = np.triu(np.random.randn(n, n), k=-1).astype(float, order='F')
    B = np.triu(np.random.randn(n, n)).astype(float, order='F')

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        'S', 'I', 'I', n, ilo, ihi, A, B
    )

    assert info == 0

    assert S.shape == (n, n)
    assert T.shape == (n, n)
    assert Q.shape == (n, n)
    assert Z.shape == (n, n)
    assert alphar.shape == (n,)
    assert alphai.shape == (n,)
    assert beta.shape == (n,)

    np.testing.assert_allclose(np.dot(Q.T, Q), np.eye(n), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.dot(Z.T, Z), np.eye(n), rtol=1e-13, atol=1e-14)


def test_mb03xp_eigenvalue_only():
    """
    Test MB03XP with JOB='E' (eigenvalues only).

    Random seed: 42 (for reproducibility)
    """
    from slicot import mb03xp

    np.random.seed(42)
    n = 4
    ilo = 1
    ihi = 4

    A = np.triu(np.random.randn(n, n), k=-1).astype(float, order='F')
    B = np.triu(np.random.randn(n, n)).astype(float, order='F')

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        'E', 'N', 'N', n, ilo, ihi, A, B
    )

    assert info == 0
    assert alphar.shape == (n,)
    assert alphai.shape == (n,)
    assert beta.shape == (n,)


def test_mb03xp_quick_return_n_zero():
    """Test quick return when N=0."""
    from slicot import mb03xp

    n = 0
    ilo = 1
    ihi = 0

    A = np.zeros((1, 1), order='F')
    B = np.zeros((1, 1), order='F')

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        'S', 'I', 'I', n, ilo, ihi, A, B
    )

    assert info == 0


def test_mb03xp_quick_return_ilo_eq_ihi_plus_1():
    """Test quick return when ILO=IHI+1 (no active block)."""
    from slicot import mb03xp

    n = 4
    ilo = 5
    ihi = 4

    A = np.eye(n, order='F')
    B = np.eye(n, order='F')

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        'S', 'I', 'I', n, ilo, ihi, A, B
    )

    assert info == 0


def test_mb03xp_n5_transformation():
    """
    Test MB03XP with N=5 case - verify shapes and orthogonality.

    Random seed: 888 (for reproducibility)
    """
    from slicot import mb03xp

    np.random.seed(888)
    n = 5
    ilo = 1
    ihi = 5

    A = np.triu(np.random.randn(n, n), k=-1).astype(float, order='F')
    B = np.triu(np.random.randn(n, n)).astype(float, order='F')

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        'S', 'I', 'I', n, ilo, ihi, A, B
    )

    assert info == 0

    assert S.shape == (n, n)
    assert T.shape == (n, n)
    assert Q.shape == (n, n)
    assert Z.shape == (n, n)
    assert alphar.shape == (n,)
    assert alphai.shape == (n,)
    assert beta.shape == (n,)

    np.testing.assert_allclose(np.dot(Q.T, Q), np.eye(n), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.dot(Z.T, Z), np.eye(n), rtol=1e-13, atol=1e-14)


def test_mb03xp_negative_n():
    """Test error for negative N."""
    from slicot import mb03xp

    n = -1
    ilo = 1
    ihi = 0

    A = np.zeros((1, 1), order='F')
    B = np.zeros((1, 1), order='F')

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        'S', 'I', 'I', n, ilo, ihi, A, B
    )

    assert info == -4


def test_mb03xp_invalid_job():
    """Test error for invalid JOB parameter."""
    from slicot import mb03xp

    n = 4
    ilo = 1
    ihi = 4

    A = np.eye(n, order='F')
    B = np.eye(n, order='F')

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        'X', 'I', 'I', n, ilo, ihi, A, B
    )

    assert info == -1


def test_mb03xp_invalid_ilo():
    """Test error for invalid ILO."""
    from slicot import mb03xp

    n = 4
    ilo = 0
    ihi = 4

    A = np.eye(n, order='F')
    B = np.eye(n, order='F')

    S, T, Q, Z, alphar, alphai, beta, info = mb03xp(
        'S', 'I', 'I', n, ilo, ihi, A, B
    )

    assert info == -5
