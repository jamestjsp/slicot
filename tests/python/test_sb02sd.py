"""
Tests for SB02SD: Condition and error bound for discrete-time Riccati equation.

SB02SD estimates conditioning and computes error bound for:
    X = op(A)'*X*(I_n + G*X)^-1*op(A) + Q
where op(A) = A or A' and Q, G are symmetric.
"""

import numpy as np
import pytest
from slicot import sb02sd


def test_sb02sd_rcond_only():
    """
    Test reciprocal condition number estimation with JOB='C'.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 3

    a = np.random.randn(n, n).astype(float, order='F')
    for i in range(n):
        a[i, i] = 0.5 + 0.1 * i

    t = np.zeros((n, n), order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)

    g = np.eye(n, order='F', dtype=float) * 0.1
    q = np.eye(n, order='F', dtype=float) * 0.5
    x = np.eye(n, order='F', dtype=float) * 0.3

    t_out, u_out, sepd, rcond, ferr, info = sb02sd(
        'C', 'N', 'N', 'U', 'O', n, a, t, u, g, q, x)

    assert info >= 0
    assert rcond > 0.0


def test_sb02sd_ferr_only():
    """
    Test forward error estimation with JOB='E'.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 3

    a = np.random.randn(n, n).astype(float, order='F')
    for i in range(n):
        a[i, i] = 0.5 + 0.1 * i

    t = np.zeros((n, n), order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)

    g = np.eye(n, order='F', dtype=float) * 0.1
    q = np.eye(n, order='F', dtype=float) * 0.5
    x = np.eye(n, order='F', dtype=float) * 0.3

    t_out, u_out, sepd, rcond, ferr, info = sb02sd(
        'E', 'N', 'N', 'U', 'O', n, a, t, u, g, q, x)

    assert info >= 0
    assert ferr >= 0.0


def test_sb02sd_both():
    """
    Test computing both with JOB='B'.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 3

    a = np.random.randn(n, n).astype(float, order='F')
    for i in range(n):
        a[i, i] = 0.5 + 0.1 * i

    t = np.zeros((n, n), order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)

    g = np.eye(n, order='F', dtype=float) * 0.1
    q = np.eye(n, order='F', dtype=float) * 0.5
    x = np.eye(n, order='F', dtype=float) * 0.3

    t_out, u_out, sepd, rcond, ferr, info = sb02sd(
        'B', 'N', 'N', 'U', 'O', n, a, t, u, g, q, x)

    assert info >= 0
    assert rcond > 0.0
    assert ferr >= 0.0


def test_sb02sd_fact_supplied():
    """
    Test with FACT='F' (Schur factorization supplied).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 3

    a = np.random.randn(n, n).astype(float, order='F')
    for i in range(n):
        a[i, i] = 0.5 + 0.1 * i

    t = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        t[i, i] = 0.4 + 0.15 * i
        for j in range(i + 1, n):
            t[i, j] = np.random.randn() * 0.1

    u = np.eye(n, order='F', dtype=float)

    g = np.eye(n, order='F', dtype=float) * 0.1
    q = np.eye(n, order='F', dtype=float) * 0.5
    x = np.eye(n, order='F', dtype=float) * 0.3

    t_out, u_out, sepd, rcond, ferr, info = sb02sd(
        'C', 'F', 'N', 'U', 'R', n, a, t, u, g, q, x)

    assert info >= 0


def test_sb02sd_trans():
    """
    Test with TRANA='T'.

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 3

    a = np.random.randn(n, n).astype(float, order='F')
    for i in range(n):
        a[i, i] = 0.5 + 0.1 * i

    t = np.zeros((n, n), order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)

    g = np.eye(n, order='F', dtype=float) * 0.1
    q = np.eye(n, order='F', dtype=float) * 0.5
    x = np.eye(n, order='F', dtype=float) * 0.3

    t_out, u_out, sepd, rcond, ferr, info = sb02sd(
        'C', 'N', 'T', 'U', 'O', n, a, t, u, g, q, x)

    assert info >= 0


def test_sb02sd_lower():
    """
    Test with UPLO='L'.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    n = 3

    a = np.random.randn(n, n).astype(float, order='F')
    for i in range(n):
        a[i, i] = 0.5 + 0.1 * i

    t = np.zeros((n, n), order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)

    g = np.eye(n, order='F', dtype=float) * 0.1
    q = np.eye(n, order='F', dtype=float) * 0.5
    x = np.eye(n, order='F', dtype=float) * 0.3

    t_out, u_out, sepd, rcond, ferr, info = sb02sd(
        'C', 'N', 'N', 'L', 'O', n, a, t, u, g, q, x)

    assert info >= 0


def test_sb02sd_reduced():
    """
    Test with LYAPUN='R' (reduced form).

    Random seed: 444 (for reproducibility)
    """
    np.random.seed(444)
    n = 3

    a = np.random.randn(n, n).astype(float, order='F')
    for i in range(n):
        a[i, i] = 0.5 + 0.1 * i

    t = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        t[i, i] = 0.4 + 0.15 * i
        for j in range(i + 1, n):
            t[i, j] = np.random.randn() * 0.1

    u = np.eye(n, order='F', dtype=float)

    g = np.eye(n, order='F', dtype=float) * 0.1
    q = np.eye(n, order='F', dtype=float) * 0.5
    x = np.eye(n, order='F', dtype=float) * 0.3

    t_out, u_out, sepd, rcond, ferr, info = sb02sd(
        'B', 'F', 'N', 'U', 'R', n, a, t, u, g, q, x)

    assert info >= 0


def test_sb02sd_n_zero():
    """
    Test quick return for n=0
    """
    n = 0
    a = np.array([], order='F', dtype=float).reshape(0, 0)
    t = np.array([], order='F', dtype=float).reshape(0, 0)
    u = np.array([], order='F', dtype=float).reshape(0, 0)
    g = np.array([], order='F', dtype=float).reshape(0, 0)
    q = np.array([], order='F', dtype=float).reshape(0, 0)
    x = np.array([], order='F', dtype=float).reshape(0, 0)

    t_out, u_out, sepd, rcond, ferr, info = sb02sd(
        'B', 'N', 'N', 'U', 'O', n, a, t, u, g, q, x)

    assert info == 0
    assert rcond == 1.0
    assert ferr == 0.0


def test_sb02sd_invalid_job():
    """
    Test error handling for invalid JOB parameter
    """
    n = 2
    a = np.eye(n, order='F', dtype=float)
    t = np.eye(n, order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)
    g = np.eye(n, order='F', dtype=float)
    q = np.eye(n, order='F', dtype=float)
    x = np.eye(n, order='F', dtype=float)

    t_out, u_out, sepd, rcond, ferr, info = sb02sd(
        'X', 'N', 'N', 'U', 'O', n, a, t, u, g, q, x)

    assert info == -1
