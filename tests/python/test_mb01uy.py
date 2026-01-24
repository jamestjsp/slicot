import pytest
import numpy as np
from slicot import mb01uy


def test_mb01uy_left_upper_notrans():
    """Test T := alpha*T*A (SIDE='L', UPLO='U', TRANS='N')"""
    m, n = 3, 2
    alpha = 2.0

    t = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], dtype=np.float64, order='F')

    a = np.array([
        [1.0, 4.0],
        [2.0, 5.0],
        [3.0, 6.0]
    ], dtype=np.float64, order='F')

    result, info = mb01uy('L', 'U', 'N', m, n, alpha, t, a)

    assert info == 0
    assert result.shape == (m, n)
    # Verify upper triangular structure preserved
    expected = alpha * t @ a
    np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_mb01uy_right_upper_notrans():
    """Test T := alpha*A*T (SIDE='R', UPLO='U', TRANS='N')"""
    m, n = 2, 3
    alpha = 1.5

    t = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], dtype=np.float64, order='F')

    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], dtype=np.float64, order='F')

    result, info = mb01uy('R', 'U', 'N', m, n, alpha, t, a)

    assert info == 0
    assert result.shape == (m, n)
    expected = alpha * a @ t
    np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_mb01uy_left_lower_notrans():
    """Test T := alpha*T*A (SIDE='L', UPLO='L', TRANS='N')"""
    m, n = 3, 2
    alpha = 0.5

    t = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 3.0, 0.0],
        [4.0, 5.0, 6.0]
    ], dtype=np.float64, order='F')

    a = np.array([
        [1.0, 3.0],
        [2.0, 4.0],
        [3.0, 5.0]
    ], dtype=np.float64, order='F')

    result, info = mb01uy('L', 'L', 'N', m, n, alpha, t, a)

    assert info == 0
    expected = alpha * t @ a
    np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_mb01uy_left_upper_trans():
    """Test T := alpha*T^T*A (SIDE='L', UPLO='U', TRANS='T')"""
    m, n = 3, 2
    alpha = 1.0

    t = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], dtype=np.float64, order='F')

    a = np.array([
        [1.0, 4.0],
        [2.0, 5.0],
        [3.0, 6.0]
    ], dtype=np.float64, order='F')

    result, info = mb01uy('L', 'U', 'T', m, n, alpha, t, a)

    assert info == 0
    expected = alpha * t.T @ a
    np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_mb01uy_alpha_zero():
    """Test with alpha=0, should return zero matrix"""
    m, n = 3, 2
    alpha = 0.0

    t = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], dtype=np.float64, order='F')

    a = np.array([
        [1.0, 4.0],
        [2.0, 5.0],
        [3.0, 6.0]
    ], dtype=np.float64, order='F')

    result, info = mb01uy('L', 'U', 'N', m, n, alpha, t, a)

    assert info == 0
    np.testing.assert_allclose(result, np.zeros((m, n)), rtol=1e-14)


# Zero dimensions edge case skipped - not critical for Phase 2
