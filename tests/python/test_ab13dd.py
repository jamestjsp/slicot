"""
Tests for AB13DD: L-infinity norm of state-space system.

Computes ||G(lambda)||_inf where G(lambda) = C*inv(lambda*E - A)*B + D.

Uses numpy only.
"""

import numpy as np


def test_ab13dd_html_example():
    """
    Test AB13DD with example from HTML documentation.

    N=6, M=1, P=1, DICO='C', JOBE='I', JOBD='D'
    Expected: L-infinity norm ~ 5e5, peak frequency ~ 1.414
    """
    from slicot import ab13dd

    n, m, p = 6, 1, 1

    a = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [-0.5, -0.0002, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -0.00002, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, -2.0, -0.000002]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    b = np.array([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]], order='F', dtype=float)

    c = np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]], order='F', dtype=float)

    d = np.array([[0.0]], order='F', dtype=float)

    fpeak = np.array([0.0, 1.0], order='F', dtype=float)
    tol = 1e-9

    gpeak, fpeak_out, info = ab13dd(
        'C', 'I', 'N', 'D', n, m, p, fpeak, a, e, b, c, d, tol
    )

    assert info == 0
    np.testing.assert_allclose(gpeak[0], 5.0e5, rtol=0.01)
    np.testing.assert_allclose(fpeak_out[0], 1.414, rtol=0.01)


def test_ab13dd_continuous_no_d():
    """
    Test AB13DD with continuous-time system, D=0.

    Random seed: 42 (for reproducibility)
    """
    from slicot import ab13dd

    np.random.seed(42)
    n, m, p = 3, 1, 1

    a = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, -2.0, 0.0],
        [0.0, 0.0, -3.0]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)
    b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
    d = np.zeros((p, m), order='F', dtype=float)

    fpeak = np.array([0.0, 1.0], order='F', dtype=float)
    tol = 1e-9

    gpeak, fpeak_out, info = ab13dd(
        'C', 'I', 'N', 'Z', n, m, p, fpeak, a, e, b, c, d, tol
    )

    assert info >= 0
    assert gpeak[0] > 0
    assert gpeak[1] >= 0


def test_ab13dd_discrete():
    """
    Test AB13DD with discrete-time system.

    Random seed: 123 (for reproducibility)
    """
    from slicot import ab13dd

    np.random.seed(123)
    n, m, p = 2, 1, 1

    a = np.array([
        [0.5, 0.0],
        [0.0, 0.3]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)
    b = np.array([[1.0], [0.5]], order='F', dtype=float)
    c = np.array([[1.0, 0.5]], order='F', dtype=float)
    d = np.array([[0.1]], order='F', dtype=float)

    fpeak = np.array([0.0, 1.0], order='F', dtype=float)
    tol = 1e-9

    gpeak, fpeak_out, info = ab13dd(
        'D', 'I', 'N', 'D', n, m, p, fpeak, a, e, b, c, d, tol
    )

    assert info >= 0
    assert gpeak[0] > 0


def test_ab13dd_descriptor():
    """
    Test AB13DD with descriptor system (JOBE='G').

    Random seed: 456 (for reproducibility)
    """
    from slicot import ab13dd

    np.random.seed(456)
    n, m, p = 2, 1, 1

    a = np.array([
        [-1.0, 0.0],
        [0.0, -2.0]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 0.1],
        [0.0, 1.0]
    ], order='F', dtype=float)

    b = np.array([[1.0], [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 1.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    fpeak = np.array([0.0, 1.0], order='F', dtype=float)
    tol = 1e-9

    gpeak, fpeak_out, info = ab13dd(
        'C', 'G', 'N', 'Z', n, m, p, fpeak, a, e, b, c, d, tol
    )

    assert info >= 0
    assert gpeak[0] > 0


def test_ab13dd_n0():
    """Test AB13DD with n=0 (quick return)."""
    from slicot import ab13dd

    n, m, p = 0, 1, 1

    a = np.zeros((1, 1), order='F', dtype=float)
    e = np.zeros((1, 1), order='F', dtype=float)
    b = np.zeros((1, 1), order='F', dtype=float)
    c = np.zeros((1, 1), order='F', dtype=float)
    d = np.array([[1.0]], order='F', dtype=float)

    fpeak = np.array([0.0, 1.0], order='F', dtype=float)
    tol = 1e-9

    gpeak, fpeak_out, info = ab13dd(
        'C', 'I', 'N', 'D', n, m, p, fpeak, a, e, b, c, d, tol
    )

    assert info == 0
    np.testing.assert_allclose(gpeak[0], 1.0, rtol=1e-14)


def test_ab13dd_m0_p0():
    """Test AB13DD with m=0 or p=0 (quick return)."""
    from slicot import ab13dd

    n, m, p = 2, 0, 1

    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.zeros((n, 1), order='F', dtype=float)
    c = np.zeros((1, n), order='F', dtype=float)
    d = np.zeros((1, 1), order='F', dtype=float)

    fpeak = np.array([0.0, 1.0], order='F', dtype=float)
    tol = 1e-9

    gpeak, fpeak_out, info = ab13dd(
        'C', 'I', 'N', 'D', n, m, p, fpeak, a, e, b, c, d, tol
    )

    assert info == 0
    np.testing.assert_allclose(gpeak[0], 0.0, atol=1e-14)


def test_ab13dd_invalid_dico():
    """Test AB13DD with invalid DICO parameter."""
    from slicot import ab13dd

    n, m, p = 2, 1, 1

    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)
    d = np.zeros((p, m), order='F', dtype=float)

    fpeak = np.array([0.0, 1.0], order='F', dtype=float)
    tol = 1e-9

    gpeak, fpeak_out, info = ab13dd(
        'X', 'I', 'N', 'D', n, m, p, fpeak, a, e, b, c, d, tol
    )

    assert info == -1
