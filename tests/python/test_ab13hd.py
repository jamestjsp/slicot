"""
Tests for AB13HD: L-infinity norm of standard/descriptor state-space system.

AB13HD computes the L-infinity norm of a proper continuous-time or
causal discrete-time system, either standard or in the descriptor form:

    G(lambda) = C * (lambda*E - A)^(-1) * B + D

Uses numpy only.
"""

import numpy as np


def test_ab13hd_continuous_standard_basic():
    """
    Test AB13HD with continuous-time standard system (JOBE='I').

    Uses the same data as AB13DD HTML example.
    N=6, M=1, P=1, DICO='C', JOBE='I', JOBD='D'
    Expected: L-infinity norm ~ 5e5, peak frequency ~ 1.414
    """
    from slicot import ab13hd

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
    tol = np.array([1e-9, -1.0], order='F', dtype=float)

    gpeak, fpeak_out, nr, iwarn, info = ab13hd(
        'C', 'I', 'N', 'D', 'N', 'N', 'A',
        n, m, p, 0, fpeak, a, e, b, c, d, tol
    )

    assert info == 0
    assert iwarn == 0
    assert nr == n
    np.testing.assert_allclose(gpeak[0], 5.0e5, rtol=0.01)
    np.testing.assert_allclose(fpeak_out[0], 1.414, rtol=0.01)


def test_ab13hd_continuous_no_d():
    """
    Test AB13HD with continuous-time standard system, D=0.

    Random seed: 42 (for reproducibility)
    """
    from slicot import ab13hd

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
    tol = np.array([1e-9, -1.0], order='F', dtype=float)

    gpeak, fpeak_out, nr, iwarn, info = ab13hd(
        'C', 'I', 'N', 'Z', 'N', 'N', 'A',
        n, m, p, 0, fpeak, a, e, b, c, d, tol
    )

    assert info >= 0
    assert gpeak[0] > 0
    assert gpeak[1] >= 0


def test_ab13hd_discrete():
    """
    Test AB13HD with discrete-time standard system.

    Random seed: 123 (for reproducibility)
    """
    from slicot import ab13hd

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
    tol = np.array([1e-9, -1.0], order='F', dtype=float)

    gpeak, fpeak_out, nr, iwarn, info = ab13hd(
        'D', 'I', 'N', 'D', 'N', 'N', 'A',
        n, m, p, 0, fpeak, a, e, b, c, d, tol
    )

    assert info >= 0
    assert gpeak[0] > 0


def test_ab13hd_descriptor():
    """
    Test AB13HD with general descriptor system (JOBE='G').

    Note: JOBE='G' not yet implemented in C translation - returns info=-2.
    Random seed: 456 (for reproducibility)
    """
    from slicot import ab13hd

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
    tol = np.array([1e-9, -1.0], order='F', dtype=float)

    gpeak, fpeak_out, nr, iwarn, info = ab13hd(
        'C', 'G', 'N', 'Z', 'N', 'N', 'A',
        n, m, p, n, fpeak, a, e, b, c, d, tol
    )

    assert info == -2


def test_ab13hd_n0():
    """Test AB13HD with n=0 (quick return). Returns gpeak=0."""
    from slicot import ab13hd

    n, m, p = 0, 1, 1

    a = np.zeros((1, 1), order='F', dtype=float)
    e = np.zeros((1, 1), order='F', dtype=float)
    b = np.zeros((1, 1), order='F', dtype=float)
    c = np.zeros((1, 1), order='F', dtype=float)
    d = np.array([[1.0]], order='F', dtype=float)

    fpeak = np.array([0.0, 1.0], order='F', dtype=float)
    tol = np.array([1e-9, -1.0], order='F', dtype=float)

    gpeak, fpeak_out, nr, iwarn, info = ab13hd(
        'C', 'I', 'N', 'D', 'N', 'N', 'A',
        n, m, p, 0, fpeak, a, e, b, c, d, tol
    )

    assert info == 0
    np.testing.assert_allclose(gpeak[0], 0.0, atol=1e-14)


def test_ab13hd_m0_p0():
    """Test AB13HD with m=0 (quick return)."""
    from slicot import ab13hd

    n, m, p = 2, 0, 1

    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.zeros((n, 1), order='F', dtype=float)
    c = np.zeros((1, n), order='F', dtype=float)
    d = np.zeros((1, 1), order='F', dtype=float)

    fpeak = np.array([0.0, 1.0], order='F', dtype=float)
    tol = np.array([1e-9, -1.0], order='F', dtype=float)

    gpeak, fpeak_out, nr, iwarn, info = ab13hd(
        'C', 'I', 'N', 'D', 'N', 'N', 'A',
        n, m, p, 0, fpeak, a, e, b, c, d, tol
    )

    assert info == 0
    np.testing.assert_allclose(gpeak[0], 0.0, atol=1e-14)


def test_ab13hd_invalid_dico():
    """Test AB13HD with invalid DICO parameter."""
    from slicot import ab13hd

    n, m, p = 2, 1, 1

    a = np.eye(n, order='F', dtype=float)
    e = np.eye(n, order='F', dtype=float)
    b = np.ones((n, m), order='F', dtype=float)
    c = np.ones((p, n), order='F', dtype=float)
    d = np.zeros((p, m), order='F', dtype=float)

    fpeak = np.array([0.0, 1.0], order='F', dtype=float)
    tol = np.array([1e-9, -1.0], order='F', dtype=float)

    gpeak, fpeak_out, nr, iwarn, info = ab13hd(
        'X', 'I', 'N', 'D', 'N', 'N', 'A',
        n, m, p, 0, fpeak, a, e, b, c, d, tol
    )

    assert info == -1


def test_ab13hd_fullrd_option():
    """
    Test AB13HD with JOBD='F' (D full rank, continuous standard).

    Random seed: 789 (for reproducibility)
    """
    from slicot import ab13hd

    np.random.seed(789)
    n, m, p = 2, 1, 1

    a = np.array([
        [-1.0, 0.0],
        [0.0, -2.0]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)
    b = np.array([[1.0], [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 1.0]], order='F', dtype=float)
    d = np.array([[0.5]], order='F', dtype=float)

    fpeak = np.array([0.0, 1.0], order='F', dtype=float)
    tol = np.array([1e-9, -1.0], order='F', dtype=float)

    gpeak, fpeak_out, nr, iwarn, info = ab13hd(
        'C', 'I', 'N', 'F', 'N', 'N', 'A',
        n, m, p, 0, fpeak, a, e, b, c, d, tol
    )

    assert info >= 0
    assert gpeak[0] > 0


def test_ab13hd_equilibration():
    """
    Test AB13HD with equilibration option (EQUIL='S').

    Random seed: 321 (for reproducibility)
    """
    from slicot import ab13hd

    np.random.seed(321)
    n, m, p = 3, 1, 1

    a = np.array([
        [-1.0, 0.1, 0.0],
        [0.0, -2.0, 0.1],
        [0.0, 0.0, -3.0]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)
    b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
    c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    fpeak = np.array([0.0, 1.0], order='F', dtype=float)
    tol = np.array([1e-9, 0.01], order='F', dtype=float)

    gpeak, fpeak_out, nr, iwarn, info = ab13hd(
        'C', 'I', 'S', 'Z', 'N', 'N', 'A',
        n, m, p, 0, fpeak, a, e, b, c, d, tol
    )

    assert info >= 0
    assert gpeak[0] > 0


def test_ab13hd_poles_partial():
    """
    Test AB13HD with POLES='P' (use partial poles).

    Random seed: 555 (for reproducibility)
    """
    from slicot import ab13hd

    np.random.seed(555)
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
    tol = np.array([1e-9, -1.0], order='F', dtype=float)

    gpeak, fpeak_out, nr, iwarn, info = ab13hd(
        'C', 'I', 'N', 'Z', 'N', 'N', 'P',
        n, m, p, 0, fpeak, a, e, b, c, d, tol
    )

    assert info >= 0
    assert gpeak[0] > 0


def test_ab13hd_compressed_e():
    """
    Test AB13HD with compressed descriptor matrix (JOBE='C').

    Note: JOBE='C' not yet implemented in C translation - returns info=-2.
    Random seed: 777 (for reproducibility)
    """
    from slicot import ab13hd

    np.random.seed(777)
    n, m, p = 3, 1, 1
    ranke = 2

    a = np.array([
        [-1.0, 0.1, 0.0],
        [0.0, -2.0, 0.1],
        [0.0, 0.0, -1.0]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    b = np.array([[1.0], [1.0], [0.5]], order='F', dtype=float)
    c = np.array([[1.0, 1.0, 0.5]], order='F', dtype=float)
    d = np.array([[0.0]], order='F', dtype=float)

    fpeak = np.array([0.0, 1.0], order='F', dtype=float)
    tol = np.array([1e-9, -1.0], order='F', dtype=float)

    gpeak, fpeak_out, nr, iwarn, info = ab13hd(
        'C', 'C', 'N', 'Z', 'N', 'N', 'A',
        n, m, p, ranke, fpeak, a, e, b, c, d, tol
    )

    assert info == -2


def test_ab13hd_hinf_norm_mimo():
    """
    Test AB13HD with MIMO system (multiple inputs/outputs).

    Random seed: 999 (for reproducibility)
    """
    from slicot import ab13hd

    np.random.seed(999)
    n, m, p = 2, 2, 2

    a = np.array([
        [-1.0, 0.0],
        [0.0, -2.0]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)
    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)
    c = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)
    d = np.zeros((p, m), order='F', dtype=float)

    fpeak = np.array([0.0, 1.0], order='F', dtype=float)
    tol = np.array([1e-9, -1.0], order='F', dtype=float)

    gpeak, fpeak_out, nr, iwarn, info = ab13hd(
        'C', 'I', 'N', 'Z', 'N', 'N', 'A',
        n, m, p, 0, fpeak, a, e, b, c, d, tol
    )

    assert info >= 0
    assert gpeak[0] > 0
    assert gpeak[1] >= 0
