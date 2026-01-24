import pytest
import numpy as np
from slicot import sg03bx


def test_sg03bx_continuous_basic():
    """Test continuous-time case with stable pencil"""
    dico = 'C'
    trans = 'N'

    # Stable pencil A - lambda*E with complex conjugate eigenvalues
    # E upper triangular, positive diagonal
    a = np.array([
        [-1.0, 2.0],
        [-2.0, -1.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    # B upper triangular
    b = np.array([
        [1.0, 0.5],
        [0.0, 0.5]
    ], dtype=np.float64, order='F')

    u, scale, m1, m2, info = sg03bx(dico, trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (2, 2)
    assert m1.shape == (2, 2)
    assert m2.shape == (2, 2)
    # U should be upper triangular
    assert u[1, 0] == 0.0
    # Main diagonal non-negative
    assert u[0, 0] >= 0.0
    assert u[1, 1] >= 0.0


def test_sg03bx_discrete_basic():
    """Test discrete-time case with stable pencil"""
    dico = 'D'
    trans = 'N'

    # Stable pencil for discrete-time (eigenvalues inside unit circle)
    # Need ||A|| < ||E||
    a = np.array([
        [0.3, 0.2],
        [-0.2, 0.3]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.1],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [0.5, 0.2],
        [0.0, 0.5]
    ], dtype=np.float64, order='F')

    u, scale, m1, m2, info = sg03bx(dico, trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0
    assert u.shape == (2, 2)
    assert u[1, 0] == 0.0
    assert u[0, 0] >= 0.0
    assert u[1, 1] >= 0.0


def test_sg03bx_transpose():
    """Test transposed equation"""
    dico = 'C'
    trans = 'T'

    a = np.array([
        [-1.0, 2.0],
        [-2.0, -1.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.5],
        [0.0, 0.5]
    ], dtype=np.float64, order='F')

    u, scale, m1, m2, info = sg03bx(dico, trans, a, e, b)

    assert info == 0
    assert 0.0 < scale <= 1.0


def test_sg03bx_error_real_eigenvalues():
    """Test error when pencil has real eigenvalues (not complex conjugate)"""
    dico = 'C'
    trans = 'N'

    # Diagonal A and E -> real eigenvalues
    a = np.array([
        [-1.0, 0.0],
        [0.0, -2.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.5],
        [0.0, 0.5]
    ], dtype=np.float64, order='F')

    u, scale, m1, m2, info = sg03bx(dico, trans, a, e, b)

    assert info == 2  # Not complex conjugate


def test_sg03bx_error_unstable():
    """Test error when eigenvalues not in correct half-plane"""
    dico = 'C'
    trans = 'N'

    # Unstable for continuous-time (eigenvalues in left half plane)
    a = np.array([
        [1.0, 2.0],
        [-2.0, 1.0]
    ], dtype=np.float64, order='F')

    e = np.array([
        [1.0, 0.5],
        [0.0, 1.0]
    ], dtype=np.float64, order='F')

    b = np.array([
        [1.0, 0.5],
        [0.0, 0.5]
    ], dtype=np.float64, order='F')

    u, scale, m1, m2, info = sg03bx(dico, trans, a, e, b)

    assert info == 3  # Stability error
