"""
Tests for SB03OD - Lyapunov equation solver computing Cholesky factor.

SB03OD solves for X = op(U)'*op(U) either the stable continuous-time Lyapunov equation:
    op(A)'*X + X*op(A) = -scale^2*op(B)'*op(B)
or the convergent discrete-time Lyapunov equation:
    op(A)'*X*op(A) - X = -scale^2*op(B)'*op(B)

where A is N-by-N, op(B) is M-by-N, U is upper triangular (Cholesky factor).
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
def sb03od(lib):
    """Get the sb03od function with proper signature."""
    func = lib.sb03od
    func.argtypes = [
        ctypes.c_char_p,                  # dico
        ctypes.c_char_p,                  # fact
        ctypes.c_char_p,                  # trans
        ctypes.c_int,                     # n
        ctypes.c_int,                     # m
        ctypes.POINTER(ctypes.c_double),  # a
        ctypes.c_int,                     # lda
        ctypes.POINTER(ctypes.c_double),  # q
        ctypes.c_int,                     # ldq
        ctypes.POINTER(ctypes.c_double),  # b
        ctypes.c_int,                     # ldb
        ctypes.POINTER(ctypes.c_double),  # scale
        ctypes.POINTER(ctypes.c_double),  # wr
        ctypes.POINTER(ctypes.c_double),  # wi
        ctypes.POINTER(ctypes.c_double),  # dwork
        ctypes.c_int,                     # ldwork
        ctypes.POINTER(ctypes.c_int),     # info
    ]
    func.restype = None
    return func


"""Test using the example from SLICOT HTML documentation."""

def test_continuous_nofact_notrans(sb03od):
    """
    Test continuous-time, compute Schur factorization, no transpose.
    
    Example data from SLICOT HTML doc for SB03OD.
    N=4, M=5, DICO='C', FACT='N', TRANS='N'
    """
    n, m = 4, 5
    
    a = np.array([
        [-1.0, 37.0, -12.0, -12.0],
        [-1.0, -10.0, 0.0, 4.0],
        [2.0, -4.0, 7.0, -6.0],
        [2.0, 2.0, 7.0, -9.0]
    ], order='F', dtype=np.float64)
    
    q = np.zeros((n, n), order='F', dtype=np.float64)
    
    b = np.array([
        [1.0, 2.5, 1.0, 3.5],
        [0.0, 1.0, 0.0, 1.0],
        [-1.0, -2.5, -1.0, -1.5],
        [1.0, 2.5, 4.0, -5.5],
        [-1.0, -2.5, -4.0, 3.5]
    ], order='F', dtype=np.float64)
    
    u_expected = np.array([
        [1.0, 3.0, 2.0, -1.0],
        [0.0, 1.0, -1.0, 1.0],
        [0.0, 0.0, 1.0, -2.0],
        [0.0, 0.0, 0.0, 1.0]
    ], order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    wr = np.zeros(n, dtype=np.float64)
    wi = np.zeros(n, dtype=np.float64)
    ldwork = max(1, 4*n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    info = ctypes.c_int(0)
    
    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.float64)
    b_padded[:m, :n] = b
    
    sb03od(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        wr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert scale.value == pytest.approx(1.0, abs=1e-4)
    
    u = np.triu(b_padded[:n, :n])
    np.testing.assert_allclose(u, u_expected, rtol=1e-3, atol=1e-4)


"""Test Lyapunov equation residual properties."""

def test_continuous_residual_verification(sb03od):
    """
    Verify A'*X + X*A = -scale^2*B'*B for continuous-time.
    
    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m = 3, 4
    
    a = -np.eye(n) - 0.5 * np.random.randn(n, n)
    a = np.asfortranarray(a)
    a_orig = a.copy()
    
    b = np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()
    
    q = np.zeros((n, n), order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    wr = np.zeros(n, dtype=np.float64)
    wi = np.zeros(n, dtype=np.float64)
    ldwork = max(1, 4*n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    info = ctypes.c_int(0)
    
    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.float64)
    b_padded[:m, :n] = b
    
    sb03od(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        wr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(b_padded[:n, :n])
    x = u.T @ u
    
    rhs = -scale.value**2 * b_orig.T @ b_orig
    residual = a_orig.T @ x + x @ a_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)

def test_discrete_residual_verification(sb03od):
    """
    Verify A'*X*A - X = -scale^2*B'*B for discrete-time.
    
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, m = 3, 4
    
    a = 0.5 * np.random.randn(n, n)
    a = np.asfortranarray(a)
    a_orig = a.copy()
    
    b = np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()
    
    q = np.zeros((n, n), order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    wr = np.zeros(n, dtype=np.float64)
    wi = np.zeros(n, dtype=np.float64)
    ldwork = max(1, 4*n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    info = ctypes.c_int(0)
    
    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.float64)
    b_padded[:m, :n] = b
    
    sb03od(
        b"D", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        wr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(b_padded[:n, :n])
    x = u.T @ u
    
    rhs = -scale.value**2 * b_orig.T @ b_orig
    residual = a_orig.T @ x @ a_orig - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


"""Test transpose option (TRANS='T')."""

def test_continuous_transpose(sb03od):
    """
    Test op(K)=K' for continuous-time.
    
    Equation: A*X + X*A' = -scale^2*B*B', where X = U*U'.
    Random seed: 200 (for reproducibility)
    """
    np.random.seed(200)
    n, m = 3, 4
    
    a = -np.eye(n) - 0.5 * np.random.randn(n, n)
    a = np.asfortranarray(a)
    a_orig = a.copy()
    
    b = np.random.randn(n, m)
    b = np.asfortranarray(b)
    b_orig = b.copy()
    
    q = np.zeros((n, n), order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    wr = np.zeros(n, dtype=np.float64)
    wi = np.zeros(n, dtype=np.float64)
    ldwork = max(1, 4*n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    info = ctypes.c_int(0)
    
    ldb = n
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.float64)
    b_padded[:n, :m] = b
    
    sb03od(
        b"C", b"N", b"T", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        wr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(b_padded[:n, :n])
    x = u @ u.T
    
    rhs = -scale.value**2 * b_orig @ b_orig.T
    residual = a_orig @ x + x @ a_orig.T - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)

def test_discrete_transpose(sb03od):
    """
    Test op(K)=K' for discrete-time.
    
    Equation: A*X*A' - X = -scale^2*B*B', where X = U*U'.
    Random seed: 201 (for reproducibility)
    """
    np.random.seed(201)
    n, m = 3, 4
    
    a = 0.5 * np.random.randn(n, n)
    a = np.asfortranarray(a)
    a_orig = a.copy()
    
    b = np.random.randn(n, m)
    b = np.asfortranarray(b)
    b_orig = b.copy()
    
    q = np.zeros((n, n), order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    wr = np.zeros(n, dtype=np.float64)
    wi = np.zeros(n, dtype=np.float64)
    ldwork = max(1, 4*n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    info = ctypes.c_int(0)
    
    ldb = n
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.float64)
    b_padded[:n, :m] = b
    
    sb03od(
        b"D", b"N", b"T", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        wr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(b_padded[:n, :n])
    x = u @ u.T
    
    rhs = -scale.value**2 * b_orig @ b_orig.T
    residual = a_orig @ x @ a_orig.T - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


"""Test FACT='F' (Schur factorization provided)."""

def test_continuous_schur_provided(sb03od):
    """
    Test with Schur factorization already provided.
    
    Random seed: 300 (for reproducibility)
    """
    np.random.seed(300)
    n, m = 3, 4
    
    s = np.array([
        [-1.0, 0.5, 0.3],
        [0.0, -2.0, 0.2],
        [0.0, 0.0, -3.0]
    ], order='F', dtype=np.float64)
    
    q = np.eye(n, order='F', dtype=np.float64)
    
    b = np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()
    
    a_orig = q @ s @ q.T
    
    scale = ctypes.c_double(0.0)
    wr = np.zeros(n, dtype=np.float64)
    wi = np.zeros(n, dtype=np.float64)
    ldwork = max(1, 4*n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    info = ctypes.c_int(0)
    
    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.float64)
    b_padded[:m, :n] = b
    
    sb03od(
        b"C", b"F", b"N", n, m,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        wr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(b_padded[:n, :n])
    x = u.T @ u
    
    rhs = -scale.value**2 * b_orig.T @ b_orig
    residual = a_orig.T @ x + x @ a_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


"""Test edge cases."""

def test_m_zero(sb03od):
    """Test M=0: U should be set to zero."""
    n, m = 3, 0
    
    a = np.array([
        [-1.0, 0.5, 0.3],
        [0.0, -2.0, 0.2],
        [0.0, 0.0, -3.0]
    ], order='F', dtype=np.float64)
    
    q = np.zeros((n, n), order='F', dtype=np.float64)
    
    b = np.zeros((n, n), order='F', dtype=np.float64)
    b[0, 0] = 999.0
    
    scale = ctypes.c_double(0.0)
    wr = np.zeros(n, dtype=np.float64)
    wi = np.zeros(n, dtype=np.float64)
    ldwork = max(1, 4*n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    info = ctypes.c_int(0)
    
    sb03od(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        wr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )
    
    assert info.value == 0
    np.testing.assert_allclose(b, 0.0, atol=1e-14)

def test_n_zero(sb03od):
    """Test N=0: quick return."""
    n, m = 0, 3
    
    a = np.zeros((1, 1), order='F', dtype=np.float64)
    q = np.zeros((1, 1), order='F', dtype=np.float64)
    b = np.zeros((m, 1), order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    wr = np.zeros(1, dtype=np.float64)
    wi = np.zeros(1, dtype=np.float64)
    dwork = np.zeros(1, dtype=np.float64)
    info = ctypes.c_int(0)
    
    sb03od(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        ctypes.byref(scale),
        wr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(info)
    )
    
    assert info.value == 0


"""Test error handling."""

def test_unstable_continuous(sb03od):
    """Test unstable A (positive eigenvalue) returns info=2."""
    n, m = 2, 2
    
    a = np.array([
        [1.0, 0.0],
        [0.0, -1.0]
    ], order='F', dtype=np.float64)
    
    q = np.zeros((n, n), order='F', dtype=np.float64)
    
    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    wr = np.zeros(n, dtype=np.float64)
    wi = np.zeros(n, dtype=np.float64)
    ldwork = max(1, 4*n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    info = ctypes.c_int(0)
    
    sb03od(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        wr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )
    
    assert info.value == 2

def test_non_convergent_discrete(sb03od):
    """Test non-convergent A (eigenvalue > 1) returns info=2."""
    n, m = 2, 2
    
    a = np.array([
        [2.0, 0.0],
        [0.0, 0.5]
    ], order='F', dtype=np.float64)
    
    q = np.zeros((n, n), order='F', dtype=np.float64)
    
    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    wr = np.zeros(n, dtype=np.float64)
    wi = np.zeros(n, dtype=np.float64)
    ldwork = max(1, 4*n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    info = ctypes.c_int(0)
    
    sb03od(
        b"D", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        wr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )
    
    assert info.value == 2

def test_invalid_dico(sb03od):
    """Test invalid DICO parameter."""
    n, m = 2, 2
    
    a = np.array([[-1.0, 0.0], [0.0, -1.0]], order='F', dtype=np.float64)
    q = np.zeros((n, n), order='F', dtype=np.float64)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    wr = np.zeros(n, dtype=np.float64)
    wi = np.zeros(n, dtype=np.float64)
    ldwork = max(1, 4*n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    info = ctypes.c_int(0)
    
    sb03od(
        b"X", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        wr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )
    
    assert info.value == -1

def test_negative_n(sb03od):
    """Test negative N."""
    n, m = -1, 2
    
    a = np.zeros((1, 1), order='F', dtype=np.float64)
    q = np.zeros((1, 1), order='F', dtype=np.float64)
    b = np.zeros((m, 1), order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    wr = np.zeros(1, dtype=np.float64)
    wi = np.zeros(1, dtype=np.float64)
    dwork = np.zeros(4, dtype=np.float64)
    info = ctypes.c_int(0)
    
    sb03od(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(scale),
        wr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 4,
        ctypes.byref(info)
    )
    
    assert info.value == -4


"""Test larger systems for robustness."""

def test_6x6_continuous(sb03od):
    """
    Test 6x6 continuous-time.

    Random seed: 600 (for reproducibility)
    """
    np.random.seed(600)
    n, m = 6, 8

    a = -np.eye(n) - 0.3 * np.random.randn(n, n)
    a = np.asfortranarray(a)
    a_orig = a.copy()

    b = np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    q = np.zeros((n, n), order='F', dtype=np.float64)

    scale = ctypes.c_double(0.0)
    wr = np.zeros(n, dtype=np.float64)
    wi = np.zeros(n, dtype=np.float64)
    ldwork = max(1, 4*n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    info = ctypes.c_int(0)

    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.float64)
    b_padded[:m, :n] = b
    
    sb03od(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        wr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(b_padded[:n, :n])
    x = u.T @ u
    
    rhs = -scale.value**2 * b_orig.T @ b_orig
    residual = a_orig.T @ x + x @ a_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)

def test_8x8_discrete_transpose(sb03od):
    """
    Test 8x8 discrete-time with transpose.

    Random seed: 800 (for reproducibility)
    """
    np.random.seed(800)
    n, m = 8, 6

    a = 0.3 * np.random.randn(n, n)
    a = np.asfortranarray(a)
    a_orig = a.copy()

    b = np.random.randn(n, m)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    q = np.zeros((n, n), order='F', dtype=np.float64)

    scale = ctypes.c_double(0.0)
    wr = np.zeros(n, dtype=np.float64)
    wi = np.zeros(n, dtype=np.float64)
    ldwork = max(1, 4*n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    info = ctypes.c_int(0)

    ldb = n
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.float64)
    b_padded[:n, :m] = b
    
    sb03od(
        b"D", b"N", b"T", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        wr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(b_padded[:n, :n])
    x = u @ u.T
    
    rhs = -scale.value**2 * b_orig @ b_orig.T
    residual = a_orig @ x @ a_orig.T - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)
