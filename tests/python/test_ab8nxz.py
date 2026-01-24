#!/usr/bin/env python3
"""
pytest tests for AB8NXZ - Extract reduced system with same transmission zeros (complex).

AB8NXZ extracts from the (N+P)-by-(M+N) system [ B  A ; D  C ] a reduced system
[ B'  A' ; D'  C' ] having the same transmission zeros but with D' of full row rank.
"""
import pytest
import numpy as np
from slicot import ab8nxz


def test_ab8nxz_basic_system():
    """Test AB8NXZ with basic SISO system.

    Random seed: 42 (for reproducibility)
    Simple state-space system with n=2, m=1, p=1.
    """
    n, m, p = 2, 1, 1
    ro = p
    sigma = 0
    svlmax = 0.0
    tol = 1.0e-10

    np.random.seed(42)
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    B = np.random.randn(n, m) + 1j * np.random.randn(n, m)
    C = np.random.randn(p, n) + 1j * np.random.randn(p, n)
    D = np.random.randn(p, m) + 1j * np.random.randn(p, m)

    abcd = np.zeros((n + p, m + n), dtype=complex, order='F')
    abcd[:n, :m] = B
    abcd[:n, m:] = A
    abcd[n:, :m] = D
    abcd[n:, m:] = C

    result = ab8nxz(n, m, p, ro, sigma, svlmax, abcd.copy(), tol)
    abcd_out, ro_out, sigma_out, mu, nu, ninfz, infz, kronl, info = result

    assert info == 0
    assert mu >= 0
    assert nu >= 0
    assert ninfz >= 0


def test_ab8nxz_mimo_system():
    """Test AB8NXZ with MIMO system.

    Random seed: 123 (for reproducibility)
    System with n=3, m=2, p=2.
    """
    n, m, p = 3, 2, 2
    ro = p
    sigma = 0
    svlmax = 0.0
    tol = 1.0e-10

    np.random.seed(123)
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    B = np.random.randn(n, m) + 1j * np.random.randn(n, m)
    C = np.random.randn(p, n) + 1j * np.random.randn(p, n)
    D = np.random.randn(p, m) + 1j * np.random.randn(p, m)

    abcd = np.zeros((n + p, m + n), dtype=complex, order='F')
    abcd[:n, :m] = B
    abcd[:n, m:] = A
    abcd[n:, :m] = D
    abcd[n:, m:] = C

    result = ab8nxz(n, m, p, ro, sigma, svlmax, abcd.copy(), tol)
    abcd_out, ro_out, sigma_out, mu, nu, ninfz, infz, kronl, info = result

    assert info == 0
    assert mu >= 0
    assert nu >= 0


def test_ab8nxz_zero_d_matrix():
    """Test AB8NXZ with D=0 (strictly proper system).

    Random seed: 456 (for reproducibility)
    """
    n, m, p = 3, 1, 2
    ro = p
    sigma = 0
    svlmax = 0.0
    tol = 1.0e-10

    np.random.seed(456)
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    B = np.random.randn(n, m) + 1j * np.random.randn(n, m)
    C = np.random.randn(p, n) + 1j * np.random.randn(p, n)
    D = np.zeros((p, m), dtype=complex)

    abcd = np.zeros((n + p, m + n), dtype=complex, order='F')
    abcd[:n, :m] = B
    abcd[:n, m:] = A
    abcd[n:, :m] = D
    abcd[n:, m:] = C

    result = ab8nxz(n, m, p, ro, sigma, svlmax, abcd.copy(), tol)
    abcd_out, ro_out, sigma_out, mu, nu, ninfz, infz, kronl, info = result

    assert info == 0


def test_ab8nxz_pertransposed_system():
    """Test AB8NXZ with pertransposed system initialization.

    Random seed: 789 (for reproducibility)
    Use ro=max(p-m,0) and sigma=m for pertransposed form.
    """
    n, m, p = 2, 2, 3
    ro = max(p - m, 0)
    sigma = m
    svlmax = 0.0
    tol = 1.0e-10

    np.random.seed(789)
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    B = np.random.randn(n, m) + 1j * np.random.randn(n, m)
    C = np.random.randn(p, n) + 1j * np.random.randn(p, n)
    D = np.random.randn(p, m) + 1j * np.random.randn(p, m)

    abcd = np.zeros((n + p, m + n), dtype=complex, order='F')
    abcd[:n, :m] = B
    abcd[:n, m:] = A
    abcd[n:, :m] = D
    abcd[n:, m:] = C

    result = ab8nxz(n, m, p, ro, sigma, svlmax, abcd.copy(), tol)
    abcd_out, ro_out, sigma_out, mu, nu, ninfz, infz, kronl, info = result

    assert info == 0


def test_ab8nxz_edge_case_n_zero():
    """Test AB8NXZ with n=0 (no states)."""
    n, m, p = 0, 2, 2
    ro = p
    sigma = 0
    svlmax = 0.0
    tol = 1.0e-10

    D = np.array([[1.0 + 0j, 0.5 + 0.5j], [0.5 - 0.5j, 1.0 + 0j]], dtype=complex, order='F')

    abcd = np.zeros((n + p, m + n), dtype=complex, order='F')
    abcd[:, :m] = D

    result = ab8nxz(n, m, p, ro, sigma, svlmax, abcd.copy(), tol)
    abcd_out, ro_out, sigma_out, mu, nu, ninfz, infz, kronl, info = result

    assert info == 0
    assert nu == 0


def test_ab8nxz_edge_case_m_zero():
    """Test AB8NXZ with m=0 (no inputs)."""
    n, m, p = 2, 0, 2
    ro = p
    sigma = 0
    svlmax = 0.0
    tol = 1.0e-10

    np.random.seed(111)
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    C = np.random.randn(p, n) + 1j * np.random.randn(p, n)

    abcd = np.zeros((n + p, m + n), dtype=complex, order='F')
    abcd[:n, :] = A
    abcd[n:, :] = C

    result = ab8nxz(n, m, p, ro, sigma, svlmax, abcd.copy(), tol)
    abcd_out, ro_out, sigma_out, mu, nu, ninfz, infz, kronl, info = result

    assert info == 0


def test_ab8nxz_error_negative_n():
    """Test AB8NXZ error handling: negative n."""
    n, m, p = -1, 1, 1
    ro = 1
    sigma = 0
    svlmax = 0.0
    tol = 1.0e-10

    abcd = np.zeros((2, 2), dtype=complex, order='F')

    with pytest.raises(ValueError, match="Dimensions must be non-negative"):
        ab8nxz(n, m, p, ro, sigma, svlmax, abcd, tol)


def test_ab8nxz_error_negative_m():
    """Test AB8NXZ error handling: negative m."""
    n, m, p = 1, -1, 1
    ro = 1
    sigma = 0
    svlmax = 0.0
    tol = 1.0e-10

    abcd = np.zeros((2, 2), dtype=complex, order='F')

    with pytest.raises(ValueError, match="Dimensions must be non-negative"):
        ab8nxz(n, m, p, ro, sigma, svlmax, abcd, tol)


def test_ab8nxz_error_negative_p():
    """Test AB8NXZ error handling: negative p."""
    n, m, p = 1, 1, -1
    ro = 1
    sigma = 0
    svlmax = 0.0
    tol = 1.0e-10

    abcd = np.zeros((2, 2), dtype=complex, order='F')

    with pytest.raises(ValueError, match="Dimensions must be non-negative"):
        ab8nxz(n, m, p, ro, sigma, svlmax, abcd, tol)


def test_ab8nxz_infz_array_size():
    """Test AB8NXZ infz array output size.

    Random seed: 222 (for reproducibility)
    """
    n, m, p = 3, 2, 2
    ro = p
    sigma = 0
    svlmax = 0.0
    tol = 1.0e-10

    np.random.seed(222)
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    B = np.random.randn(n, m) + 1j * np.random.randn(n, m)
    C = np.random.randn(p, n) + 1j * np.random.randn(p, n)
    D = np.random.randn(p, m) + 1j * np.random.randn(p, m)

    abcd = np.zeros((n + p, m + n), dtype=complex, order='F')
    abcd[:n, :m] = B
    abcd[:n, m:] = A
    abcd[n:, :m] = D
    abcd[n:, m:] = C

    result = ab8nxz(n, m, p, ro, sigma, svlmax, abcd.copy(), tol)
    abcd_out, ro_out, sigma_out, mu, nu, ninfz, infz, kronl, info = result

    assert info == 0
    assert len(infz) == n


def test_ab8nxz_larger_system():
    """Test AB8NXZ with larger system.

    Random seed: 333 (for reproducibility)
    """
    n, m, p = 5, 3, 3
    ro = p
    sigma = 0
    svlmax = 0.0
    tol = 1.0e-10

    np.random.seed(333)
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    B = np.random.randn(n, m) + 1j * np.random.randn(n, m)
    C = np.random.randn(p, n) + 1j * np.random.randn(p, n)
    D = np.random.randn(p, m) + 1j * np.random.randn(p, m)

    abcd = np.zeros((n + p, m + n), dtype=complex, order='F')
    abcd[:n, :m] = B
    abcd[:n, m:] = A
    abcd[n:, :m] = D
    abcd[n:, m:] = C

    result = ab8nxz(n, m, p, ro, sigma, svlmax, abcd.copy(), tol)
    abcd_out, ro_out, sigma_out, mu, nu, ninfz, infz, kronl, info = result

    assert info == 0
    assert mu >= 0
    assert nu >= 0
    assert nu <= n


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
