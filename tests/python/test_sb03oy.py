"""
Tests for SB03OY - 2x2 Lyapunov equation solver for Cholesky factor.

SB03OY solves for the Cholesky factor U of X, where op(U)'*op(U) = X, either:
- Continuous-time: op(S)'*X + X*op(S) = -ISGN*scale^2*op(R)'*op(R)
- Discrete-time: op(S)'*X*op(S) - X = -ISGN*scale^2*op(R)'*op(R)

where S is 2x2 with complex conjugate eigenvalues, R is 2x2 upper triangular.

This is an internal routine, tested via ctypes.
"""
import ctypes
import numpy as np
import os
import glob
import pytest


def find_slicot_library():
    """Find the slicot shared library."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_dirs = [
        os.path.join(project_root, 'build', 'macos-arm64-debug', 'src'),
        os.path.join(project_root, 'build', 'linux-x64-debug', 'src'),
        os.path.join(project_root, 'build', 'linux-x64-debug-sanitizers', 'src'),
        os.path.join(project_root, 'build', 'macos-arm64-release', 'src'),
        os.path.join(project_root, 'build', 'linux-x64-release', 'src'),
    ]
    for bd in build_dirs:
        if os.path.exists(bd):
            libs = glob.glob(os.path.join(bd, 'libslicot.*'))
            if libs:
                return libs[0]
    pytest.skip("Could not find libslicot shared library")


@pytest.fixture(scope='module')
def lib():
    """Load the slicot library."""
    lib_path = find_slicot_library()
    slicot = ctypes.CDLL(lib_path)
    return slicot


@pytest.fixture(scope='module')
def sb03oy(lib):
    """Get the sb03oy function with proper signature."""
    func = lib.sb03oy
    func.argtypes = [
        ctypes.c_bool,                    # discr
        ctypes.c_bool,                    # ltrans
        ctypes.c_int,                     # isgn
        ctypes.POINTER(ctypes.c_double),  # s
        ctypes.c_int,                     # lds
        ctypes.POINTER(ctypes.c_double),  # r
        ctypes.c_int,                     # ldr
        ctypes.POINTER(ctypes.c_double),  # a
        ctypes.c_int,                     # lda
        ctypes.POINTER(ctypes.c_double),  # scale
        ctypes.POINTER(ctypes.c_int),     # info
    ]
    func.restype = None
    return func


def make_stable_continuous_2x2(alpha, omega):
    """
    Create a 2x2 matrix with complex conjugate eigenvalues alpha +/- i*omega.
    For continuous-time stability: alpha < 0.
    
    Returns matrix: [[alpha, omega], [-omega, alpha]]
    which has eigenvalues alpha +/- i*omega.
    """
    return np.array([[alpha, omega], [-omega, alpha]], order='F', dtype=np.float64)


def make_stable_discrete_2x2(r, theta):
    """
    Create a 2x2 matrix with complex conjugate eigenvalues r*exp(+/-i*theta).
    For discrete-time stability (convergent): 0 < r < 1.
    
    Returns matrix: [[r*cos(theta), r*sin(theta)], [-r*sin(theta), r*cos(theta)]]
    which has eigenvalues r*exp(+/-i*theta).
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[r*c, r*s], [-r*s, r*c]], order='F', dtype=np.float64)


def test_sb03oy_continuous_basic(sb03oy):
    """
    Test continuous-time Lyapunov equation with no transpose.
    
    Solves: S'*X + X*S = -ISGN*scale^2*R'*R for Cholesky factor U where X = U'*U.
    
    Uses stable matrix S with eigenvalues -1 +/- 2i (negative real parts).
    """
    s = make_stable_continuous_2x2(-1.0, 2.0)
    s_orig = s.copy()
    r = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=np.float64)
    r_orig = r.copy()
    a = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    
    sb03oy(
        False, False, 1,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = r
    assert u[1, 0] == 0.0 or abs(u[1, 0]) < 1e-14
    
    x = u.T @ u
    
    rhs = -1 * scale.value**2 * r_orig.T @ r_orig
    residual = s_orig.T @ x + x @ s_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sb03oy_continuous_transpose(sb03oy):
    """
    Test continuous-time Lyapunov equation with transpose.
    
    Solves: S*X + X*S' = -ISGN*scale^2*R*R' for Cholesky factor U where X = U*U'.
    
    Uses stable matrix S with eigenvalues -0.5 +/- 1.5i.
    """
    s = make_stable_continuous_2x2(-0.5, 1.5)
    s_orig = s.copy()
    r = np.array([[2.0, 1.0], [0.0, 1.5]], order='F', dtype=np.float64)
    r_orig = r.copy()
    a = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    
    sb03oy(
        False, True, 1,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = r
    assert u[1, 0] == 0.0 or abs(u[1, 0]) < 1e-14
    
    x = u @ u.T
    
    rhs = -1 * scale.value**2 * r_orig @ r_orig.T
    residual = s_orig @ x + x @ s_orig.T - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sb03oy_discrete_basic(sb03oy):
    """
    Test discrete-time Lyapunov equation with no transpose.
    
    Solves: S'*X*S - X = -ISGN*scale^2*R'*R for Cholesky factor U where X = U'*U.
    
    Uses convergent matrix S with eigenvalue modulus 0.8 (< 1).
    """
    s = make_stable_discrete_2x2(0.8, np.pi/4)
    s_orig = s.copy()
    r = np.array([[1.0, 0.3], [0.0, 0.8]], order='F', dtype=np.float64)
    r_orig = r.copy()
    a = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    
    sb03oy(
        True, False, 1,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = r
    assert u[1, 0] == 0.0 or abs(u[1, 0]) < 1e-14
    
    x = u.T @ u
    
    rhs = -1 * scale.value**2 * r_orig.T @ r_orig
    residual = s_orig.T @ x @ s_orig - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sb03oy_discrete_transpose(sb03oy):
    """
    Test discrete-time Lyapunov equation with transpose.
    
    Solves: S*X*S' - X = -ISGN*scale^2*R*R' for Cholesky factor U where X = U*U'.
    
    Uses convergent matrix S with eigenvalue modulus 0.6.
    """
    s = make_stable_discrete_2x2(0.6, np.pi/3)
    s_orig = s.copy()
    r = np.array([[1.5, 0.2], [0.0, 1.2]], order='F', dtype=np.float64)
    r_orig = r.copy()
    a = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    
    sb03oy(
        True, True, 1,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = r
    assert u[1, 0] == 0.0 or abs(u[1, 0]) < 1e-14
    
    x = u @ u.T
    
    rhs = -1 * scale.value**2 * r_orig @ r_orig.T
    residual = s_orig @ x @ s_orig.T - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sb03oy_isgn_minus_one(sb03oy):
    """
    Test with ISGN=-1 for continuous-time.
    
    For ISGN=-1 and continuous: -S must be stable (eigenvalues have positive real).
    So we use S with positive real part.
    """
    s = make_stable_continuous_2x2(0.5, 1.0)
    s_orig = s.copy()
    r = np.array([[1.0, 0.4], [0.0, 0.9]], order='F', dtype=np.float64)
    r_orig = r.copy()
    a = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    
    sb03oy(
        False, False, -1,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = r
    x = u.T @ u
    
    rhs = -(-1) * scale.value**2 * r_orig.T @ r_orig
    residual = s_orig.T @ x + x @ s_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_sb03oy_output_a_upper_triangular(sb03oy):
    """
    Test that output matrix A is upper triangular.
    """
    s = make_stable_continuous_2x2(-1.0, 1.5)
    r = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=np.float64)
    a = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    
    sb03oy(
        False, False, 1,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    np.testing.assert_allclose(a[1, 0], 0.0, atol=1e-14)


def test_sb03oy_continuous_unstable_returns_info2(sb03oy):
    """
    Test that unstable S (for ISGN=1) returns info=2.
    
    For continuous-time with ISGN=1, S must be stable (negative real parts).
    Positive real parts should return info=2.
    """
    s = make_stable_continuous_2x2(0.5, 1.0)
    r = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=np.float64)
    a = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    
    sb03oy(
        False, False, 1,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        ctypes.byref(info)
    )
    
    assert info.value == 2


def test_sb03oy_real_eigenvalues_returns_info4(sb03oy):
    """
    Test that S with real eigenvalues returns info=4.
    
    S must have complex conjugate eigenvalues for this routine.
    """
    s = np.array([[1.0, 0.0], [0.0, 2.0]], order='F', dtype=np.float64)
    r = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=np.float64)
    a = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    
    sb03oy(
        False, False, 1,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        ctypes.byref(info)
    )
    
    assert info.value == 4


def test_sb03oy_discrete_not_convergent_returns_info2(sb03oy):
    """
    Test that discrete-time with non-convergent S (with ISGN=1) returns info=2.
    
    For discrete-time with ISGN=1, eigenvalue moduli must be < 1.
    """
    s = make_stable_discrete_2x2(1.2, np.pi/4)
    r = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=np.float64)
    a = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    
    sb03oy(
        True, False, 1,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        ctypes.byref(info)
    )
    
    assert info.value == 2


def test_sb03oy_b_u_relation_no_transpose(sb03oy):
    """
    Test the B*U = U*S relation for LTRANS=False.
    
    After solving, S contains B such that B*U = U*S (original S).
    """
    s = make_stable_continuous_2x2(-1.0, 2.0)
    s_orig = s.copy()
    r = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=np.float64)
    a = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    
    sb03oy(
        False, False, 1,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    
    b = s
    u = r
    
    lhs = b @ u
    rhs = u @ s_orig
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)


def test_sb03oy_a_u_relation_no_transpose(sb03oy):
    """
    Test the A*U = scale^2*R relation for LTRANS=False.
    
    After solving, A satisfies A*U/scale = scale*R.
    """
    s = make_stable_continuous_2x2(-1.0, 2.0)
    r = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=np.float64)
    r_orig = r.copy()
    a = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    
    sb03oy(
        False, False, 1,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    
    u = r
    
    lhs = a @ u
    rhs = scale.value**2 * r_orig
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)
