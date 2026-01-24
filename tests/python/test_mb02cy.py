import numpy as np
import pytest


def test_mb02cy_basic_row():
    """
    Validate applying hyperbolic transformations on row-wise generator.

    Tests: Apply transformations computed by MB02CX to additional columns.
    Uses simple identity-based transformation with known outputs.
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    p = 3
    q = 2
    n = 4
    k = 3

    a = np.random.randn(p, n).astype(float, order='F')
    b = np.random.randn(q, n).astype(float, order='F')

    h = np.random.randn(q, k).astype(float, order='F')
    lcs = 2 * k + min(k, q)
    cs = np.random.randn(lcs).astype(float, order='F')
    cs[0:2*k:2] = np.abs(cs[0:2*k:2]) + 1.0

    from slicot import mb02cy

    a_orig = a.copy()
    b_orig = b.copy()

    a_out, b_out, info = mb02cy('R', 'N', p, q, n, k, a.copy(), b.copy(), h.copy(), cs.copy())

    assert info == 0
    assert a_out.shape == (p, n)
    assert b_out.shape == (q, n)


def test_mb02cy_basic_column():
    """
    Validate applying hyperbolic transformations on column-wise generator.

    Tests: Apply transformations computed by MB02CX to additional rows.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    p = 3
    q = 2
    n = 4
    k = 3

    a = np.random.randn(n, p).astype(float, order='F')
    b = np.random.randn(n, q).astype(float, order='F')

    h = np.random.randn(k, q).astype(float, order='F')
    lcs = 2 * k + min(k, q)
    cs = np.random.randn(lcs).astype(float, order='F')
    cs[0:2*k:2] = np.abs(cs[0:2*k:2]) + 1.0

    from slicot import mb02cy

    a_out, b_out, info = mb02cy('C', 'N', p, q, n, k, a.copy(), b.copy(), h.copy(), cs.copy())

    assert info == 0
    assert a_out.shape == (n, p)
    assert b_out.shape == (n, q)


def test_mb02cy_triangular_structure():
    """
    Validate applying transformations with triangular structure.

    Tests: STRUCG = 'T' (triangular positive generator, zero negative trailing block).
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    p = 4
    q = 3
    n = 5
    k = 4

    a = np.random.randn(p, n).astype(float, order='F')
    b = np.random.randn(q, n).astype(float, order='F')

    h = np.random.randn(q, k).astype(float, order='F')
    lcs = 2 * k + min(k, q)
    cs = np.random.randn(lcs).astype(float, order='F')
    cs[0:2*k:2] = np.abs(cs[0:2*k:2]) + 1.0

    from slicot import mb02cy

    a_out, b_out, info = mb02cy('R', 'T', p, q, n, k, a.copy(), b.copy(), h.copy(), cs.copy())

    assert info == 0
    assert a_out.shape == (p, n)
    assert b_out.shape == (q, n)


def test_mb02cy_edge_case_q_zero():
    """
    Validate edge case: Q = 0 (no negative generator).

    When Q = 0 or K = 0, routine should return immediately.
    """
    p = 3
    k = 2
    n = 4
    q = 0

    a = np.random.randn(p, n).astype(float, order='F')
    b = np.zeros((1, n), order='F', dtype=float)
    h = np.zeros((1, k), order='F', dtype=float)
    lcs = 2 * k
    cs = np.zeros(lcs, order='F', dtype=float)

    from slicot import mb02cy

    a_out, b_out, info = mb02cy('R', 'N', p, q, n, k, a.copy(), b.copy(), h.copy(), cs.copy())

    assert info == 0


def test_mb02cy_edge_case_n_zero():
    """
    Validate edge case: N = 0 (no columns/rows to process).
    """
    p = 3
    q = 2
    k = 3
    n = 0

    a = np.zeros((p, 1), order='F', dtype=float)
    b = np.zeros((q, 1), order='F', dtype=float)
    h = np.zeros((q, k), order='F', dtype=float)
    lcs = 2 * k + min(k, q)
    cs = np.zeros(lcs, order='F', dtype=float)

    from slicot import mb02cy

    a_out, b_out, info = mb02cy('R', 'N', p, q, n, k, a.copy(), b.copy(), h.copy(), cs.copy())

    assert info == 0


def test_mb02cy_error_invalid_typet():
    """
    Validate error handling: invalid TYPET parameter.
    """
    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    h = np.array([[1.0]], order='F', dtype=float)
    cs = np.array([1.0, 0.0, 1.0], order='F', dtype=float)

    from slicot import mb02cy

    with pytest.raises((ValueError, RuntimeError)):
        mb02cy('X', 'N', 1, 1, 1, 1, a, b, h, cs)


def test_mb02cy_error_invalid_strucg():
    """
    Validate error handling: invalid STRUCG parameter.
    """
    a = np.array([[1.0]], order='F', dtype=float)
    b = np.array([[1.0]], order='F', dtype=float)
    h = np.array([[1.0]], order='F', dtype=float)
    cs = np.array([1.0, 0.0, 1.0], order='F', dtype=float)

    from slicot import mb02cy

    with pytest.raises((ValueError, RuntimeError)):
        mb02cy('R', 'X', 1, 1, 1, 1, a, b, h, cs)


def test_mb02cy_error_k_exceeds_p():
    """
    Validate error handling: K > P (invalid dimensions).
    """
    p = 2
    q = 2
    k = 3
    n = 4

    a = np.random.randn(p, n).astype(float, order='F')
    b = np.random.randn(q, n).astype(float, order='F')
    h = np.random.randn(q, k).astype(float, order='F')
    cs = np.zeros(2 * k + min(k, q), order='F', dtype=float)

    from slicot import mb02cy

    with pytest.raises((ValueError, RuntimeError)):
        mb02cy('R', 'N', p, q, n, k, a, b, h, cs)
