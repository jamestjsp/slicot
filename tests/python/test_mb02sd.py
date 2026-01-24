"""
Tests for MB02SD: LU factorization of upper Hessenberg matrix.

Uses numpy only.
"""

import numpy as np


def test_mb02sd_basic():
    """
    Test MB02SD with a basic upper Hessenberg matrix.

    Random seed: 42 (for reproducibility)
    """
    from slicot import mb02sd

    np.random.seed(42)
    n = 4

    h = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn()

    h_orig = h.copy()
    h_out, ipiv, info = mb02sd(n, h)

    assert info == 0
    assert h_out.shape == (n, n)
    assert ipiv.shape == (n,)


def test_mb02sd_n3():
    """
    Test MB02SD with 3x3 upper Hessenberg matrix.

    Random seed: 123 (for reproducibility)
    """
    from slicot import mb02sd

    np.random.seed(123)
    n = 3

    h = np.array([
        [2.0, 1.0, 3.0],
        [4.0, 5.0, 2.0],
        [0.0, 3.0, 1.0]
    ], order='F', dtype=float)

    h_out, ipiv, info = mb02sd(n, h)

    assert info == 0
    assert h_out.shape == (n, n)
    assert ipiv.shape == (n,)

    p = np.eye(n)
    for i in range(n):
        if ipiv[i] != i + 1:
            p[[i, ipiv[i] - 1]] = p[[ipiv[i] - 1, i]]

    l = np.eye(n)
    u = np.triu(h_out)

    for i in range(n - 1):
        l[i + 1, i] = h_out[i + 1, i]

    reconstructed = p @ l @ u


def test_mb02sd_singular():
    """
    Test MB02SD with a singular matrix (should return info > 0).
    """
    from slicot import mb02sd

    n = 3
    h = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    h_out, ipiv, info = mb02sd(n, h)

    assert info > 0


def test_mb02sd_n1():
    """Test MB02SD with 1x1 matrix."""
    from slicot import mb02sd

    n = 1
    h = np.array([[5.0]], order='F', dtype=float)

    h_out, ipiv, info = mb02sd(n, h)

    assert info == 0
    assert ipiv[0] == 1
    np.testing.assert_allclose(h_out[0, 0], 5.0, rtol=1e-14)


def test_mb02sd_n0():
    """Test MB02SD with n=0 (quick return)."""
    from slicot import mb02sd

    n = 0
    h = np.zeros((1, 1), order='F', dtype=float)

    h_out, ipiv, info = mb02sd(n, h)

    assert info == 0
