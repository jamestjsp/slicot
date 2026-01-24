"""
Tests for SB03OZ - Complex Lyapunov equation solver computing Cholesky factor.

Solves for X = op(U)^H * op(U) either the stable continuous-time Lyapunov equation:
    op(A)^H * X + X * op(A) = -scale^2 * op(B)^H * op(B)
or the convergent discrete-time Lyapunov equation:
    op(A)^H * X * op(A) - X = -scale^2 * op(B)^H * op(B)

where A is N-by-N complex, op(B) is M-by-N complex, U is upper triangular Cholesky factor.

Tests via ctypes since SB03OZ uses complex arrays.
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
def sb03oz(lib):
    """Get the sb03oz function with proper signature."""
    func = lib.sb03oz
    func.argtypes = [
        ctypes.c_char_p,                  # dico
        ctypes.c_char_p,                  # fact
        ctypes.c_char_p,                  # trans
        ctypes.c_int,                     # n
        ctypes.c_int,                     # m
        ctypes.POINTER(ctypes.c_double),  # a (complex, 2*doubles per element)
        ctypes.c_int,                     # lda
        ctypes.POINTER(ctypes.c_double),  # q (complex)
        ctypes.c_int,                     # ldq
        ctypes.POINTER(ctypes.c_double),  # b (complex)
        ctypes.c_int,                     # ldb
        ctypes.POINTER(ctypes.c_double),  # scale
        ctypes.POINTER(ctypes.c_double),  # w (complex eigenvalues)
        ctypes.POINTER(ctypes.c_double),  # dwork
        ctypes.POINTER(ctypes.c_double),  # zwork (complex workspace)
        ctypes.c_int,                     # lzwork
        ctypes.POINTER(ctypes.c_int),     # info
    ]
    func.restype = None
    return func


def make_stable_continuous_complex(n, seed=42):
    """
    Create an n-by-n complex matrix with stable eigenvalues (negative real parts).

    Random seed: {seed} (for reproducibility)
    """
    np.random.seed(seed)
    u, _ = np.linalg.qr(np.random.randn(n, n) + 1j*np.random.randn(n, n))
    eig = -np.random.rand(n) - 0.5 + 1j * np.random.randn(n) * 0.3
    a = u @ np.diag(eig) @ u.conj().T
    return np.asfortranarray(a)


def make_stable_discrete_complex(n, seed=42):
    """
    Create an n-by-n complex matrix with convergent eigenvalues (modulus < 1).

    Random seed: {seed} (for reproducibility)
    """
    np.random.seed(seed)
    u, _ = np.linalg.qr(np.random.randn(n, n) + 1j*np.random.randn(n, n))
    r = 0.3 + np.random.rand(n) * 0.5
    theta = np.random.rand(n) * 2 * np.pi
    eig = r * np.exp(1j * theta)
    a = u @ np.diag(eig) @ u.conj().T
    return np.asfortranarray(a)


def test_continuous_nofact_notrans(sb03oz):
    """
    Test continuous-time, compute Schur factorization, no transpose.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, m = 3, 4

    a = make_stable_continuous_complex(n, seed=42)
    a_orig = a.copy()

    b = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    scale = ctypes.c_double(0.0)
    w = np.zeros(n, dtype=np.complex128)
    ldwork = max(1, n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    lzwork = max(1, 2*n + max(min(n, m) - 2, 0))
    zwork = np.zeros(lzwork, dtype=np.complex128)
    info = ctypes.c_int(0)

    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:m, :n] = b

    sb03oz(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lzwork,
        ctypes.byref(info)
    )

    assert info.value == 0
    assert 0 < scale.value <= 1.0

    u = np.triu(b_padded[:n, :n])
    x = u.conj().T @ u

    rhs = -scale.value**2 * b_orig.conj().T @ b_orig
    residual = a_orig.conj().T @ x + x @ a_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_continuous_transpose(sb03oz):
    """
    Test continuous-time with transpose (TRANS='C').

    Equation: A * X + X * A^H = -scale^2 * B * B^H, X = U * U^H
    Random seed: 100 (for reproducibility)
    """
    np.random.seed(100)
    n, m = 3, 4

    a = make_stable_continuous_complex(n, seed=100)
    a_orig = a.copy()

    b = np.random.randn(n, m) + 1j * np.random.randn(n, m)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    scale = ctypes.c_double(0.0)
    w = np.zeros(n, dtype=np.complex128)
    ldwork = max(1, n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    lzwork = max(1, 2*n + max(min(n, m) - 2, 0))
    zwork = np.zeros(lzwork, dtype=np.complex128)
    info = ctypes.c_int(0)

    ldb = n
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:n, :m] = b

    sb03oz(
        b"C", b"N", b"C", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lzwork,
        ctypes.byref(info)
    )

    assert info.value == 0
    assert 0 < scale.value <= 1.0

    u = np.triu(b_padded[:n, :n])
    x = u @ u.conj().T

    rhs = -scale.value**2 * b_orig @ b_orig.conj().T
    residual = a_orig @ x + x @ a_orig.conj().T - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_discrete_nofact_notrans(sb03oz):
    """
    Test discrete-time, compute Schur factorization, no transpose.

    Equation: A^H * X * A - X = -scale^2 * B^H * B, X = U^H * U
    Random seed: 200 (for reproducibility)
    """
    np.random.seed(200)
    n, m = 3, 4

    a = make_stable_discrete_complex(n, seed=200)
    a_orig = a.copy()

    b = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    scale = ctypes.c_double(0.0)
    w = np.zeros(n, dtype=np.complex128)
    ldwork = max(1, n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    lzwork = max(1, 2*n + max(min(n, m) - 2, 0))
    zwork = np.zeros(lzwork, dtype=np.complex128)
    info = ctypes.c_int(0)

    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:m, :n] = b

    sb03oz(
        b"D", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lzwork,
        ctypes.byref(info)
    )

    assert info.value == 0
    assert 0 < scale.value <= 1.0

    u = np.triu(b_padded[:n, :n])
    x = u.conj().T @ u

    rhs = -scale.value**2 * b_orig.conj().T @ b_orig
    residual = a_orig.conj().T @ x @ a_orig - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_discrete_transpose(sb03oz):
    """
    Test discrete-time with transpose (TRANS='C').

    Equation: A * X * A^H - X = -scale^2 * B * B^H, X = U * U^H
    Random seed: 300 (for reproducibility)
    """
    np.random.seed(300)
    n, m = 3, 4

    a = make_stable_discrete_complex(n, seed=300)
    a_orig = a.copy()

    b = np.random.randn(n, m) + 1j * np.random.randn(n, m)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    scale = ctypes.c_double(0.0)
    w = np.zeros(n, dtype=np.complex128)
    ldwork = max(1, n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    lzwork = max(1, 2*n + max(min(n, m) - 2, 0))
    zwork = np.zeros(lzwork, dtype=np.complex128)
    info = ctypes.c_int(0)

    ldb = n
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:n, :m] = b

    sb03oz(
        b"D", b"N", b"C", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lzwork,
        ctypes.byref(info)
    )

    assert info.value == 0
    assert 0 < scale.value <= 1.0

    u = np.triu(b_padded[:n, :n])
    x = u @ u.conj().T

    rhs = -scale.value**2 * b_orig @ b_orig.conj().T
    residual = a_orig @ x @ a_orig.conj().T - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_schur_provided(sb03oz):
    """
    Test with Schur factorization already provided (FACT='F').

    Random seed: 400 (for reproducibility)
    """
    np.random.seed(400)
    n, m = 3, 4

    s = np.array([
        [-1.0 + 0.2j, 0.3 - 0.1j, 0.2 + 0.1j],
        [0.0 + 0j, -1.5 + 0.3j, 0.4 - 0.2j],
        [0.0 + 0j, 0.0 + 0j, -2.0 - 0.5j]
    ], order='F', dtype=np.complex128)

    q = np.eye(n, order='F', dtype=np.complex128)

    b = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    a_orig = q @ s @ q.conj().T

    scale = ctypes.c_double(0.0)
    w = np.zeros(n, dtype=np.complex128)
    ldwork = max(1, n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    lzwork = max(1, 2*n + max(min(n, m) - 2, 0))
    zwork = np.zeros(lzwork, dtype=np.complex128)
    info = ctypes.c_int(0)

    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:m, :n] = b

    sb03oz(
        b"C", b"F", b"N", n, m,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lzwork,
        ctypes.byref(info)
    )

    assert info.value == 0
    assert 0 < scale.value <= 1.0

    u = np.triu(b_padded[:n, :n])
    x = u.conj().T @ u

    rhs = -scale.value**2 * b_orig.conj().T @ b_orig
    residual = a_orig.conj().T @ x + x @ a_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_m_zero(sb03oz):
    """Test M=0: U should be set to zero."""
    n, m = 3, 0

    a = np.array([
        [-1.0 + 0.2j, 0.3 - 0.1j, 0.2 + 0.1j],
        [0.0 + 0j, -1.5 + 0.3j, 0.4 - 0.2j],
        [0.0 + 0j, 0.0 + 0j, -2.0 - 0.5j]
    ], order='F', dtype=np.complex128)

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    b = np.zeros((n, n), order='F', dtype=np.complex128)
    b[0, 0] = 999.0 + 0j

    scale = ctypes.c_double(0.0)
    w = np.zeros(n, dtype=np.complex128)
    dwork = np.zeros(n, dtype=np.float64)
    zwork = np.zeros(1, dtype=np.complex128)
    info = ctypes.c_int(0)

    sb03oz(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(info)
    )

    assert info.value == 0
    np.testing.assert_allclose(b, 0.0, atol=1e-14)


def test_n_zero(sb03oz):
    """Test N=0: quick return."""
    n, m = 0, 3

    a = np.zeros((1, 1), order='F', dtype=np.complex128)
    q = np.zeros((1, 1), order='F', dtype=np.complex128)
    b = np.zeros((m, 1), order='F', dtype=np.complex128)

    scale = ctypes.c_double(0.0)
    w = np.zeros(1, dtype=np.complex128)
    dwork = np.zeros(1, dtype=np.float64)
    zwork = np.zeros(1, dtype=np.complex128)
    info = ctypes.c_int(0)

    sb03oz(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        ctypes.byref(scale),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(info)
    )

    assert info.value == 0


def test_unstable_continuous(sb03oz):
    """Test unstable A (non-negative real eigenvalue) returns info=2."""
    n, m = 2, 2

    a = np.array([
        [1.0 + 0.2j, 0.0 + 0j],
        [0.0 + 0j, -1.0 + 0.3j]
    ], order='F', dtype=np.complex128)

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    b = np.array([
        [1.0 + 0j, 0.0 + 0j],
        [0.0 + 0j, 1.0 + 0j]
    ], order='F', dtype=np.complex128)

    scale = ctypes.c_double(0.0)
    w = np.zeros(n, dtype=np.complex128)
    ldwork = max(1, n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    lzwork = max(1, 2*n + max(min(n, m) - 2, 0))
    zwork = np.zeros(lzwork, dtype=np.complex128)
    info = ctypes.c_int(0)

    sb03oz(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lzwork,
        ctypes.byref(info)
    )

    assert info.value == 2


def test_non_convergent_discrete(sb03oz):
    """Test non-convergent A (eigenvalue modulus >= 1) returns info=2."""
    n, m = 2, 2

    a = np.array([
        [1.5 + 0.3j, 0.0 + 0j],
        [0.0 + 0j, 0.5 + 0.2j]
    ], order='F', dtype=np.complex128)

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    b = np.array([
        [1.0 + 0j, 0.0 + 0j],
        [0.0 + 0j, 1.0 + 0j]
    ], order='F', dtype=np.complex128)

    scale = ctypes.c_double(0.0)
    w = np.zeros(n, dtype=np.complex128)
    ldwork = max(1, n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    lzwork = max(1, 2*n + max(min(n, m) - 2, 0))
    zwork = np.zeros(lzwork, dtype=np.complex128)
    info = ctypes.c_int(0)

    sb03oz(
        b"D", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lzwork,
        ctypes.byref(info)
    )

    assert info.value == 2


def test_invalid_dico(sb03oz):
    """Test invalid DICO parameter."""
    n, m = 2, 2

    a = np.array([[-1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1.0 + 0j]], order='F', dtype=np.complex128)
    q = np.zeros((n, n), order='F', dtype=np.complex128)
    b = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]], order='F', dtype=np.complex128)

    scale = ctypes.c_double(0.0)
    w = np.zeros(n, dtype=np.complex128)
    ldwork = max(1, n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    lzwork = max(1, 2*n + max(min(n, m) - 2, 0))
    zwork = np.zeros(lzwork, dtype=np.complex128)
    info = ctypes.c_int(0)

    sb03oz(
        b"X", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lzwork,
        ctypes.byref(info)
    )

    assert info.value == -1


def test_negative_n(sb03oz):
    """Test negative N."""
    n, m = -1, 2

    a = np.zeros((1, 1), order='F', dtype=np.complex128)
    q = np.zeros((1, 1), order='F', dtype=np.complex128)
    b = np.zeros((m, 1), order='F', dtype=np.complex128)

    scale = ctypes.c_double(0.0)
    w = np.zeros(1, dtype=np.complex128)
    dwork = np.zeros(4, dtype=np.float64)
    zwork = np.zeros(4, dtype=np.complex128)
    info = ctypes.c_int(0)

    sb03oz(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(scale),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 4,
        ctypes.byref(info)
    )

    assert info.value == -4


def test_5x5_continuous(sb03oz):
    """
    Test larger 5x5 continuous-time.

    Random seed: 500 (for reproducibility)
    """
    np.random.seed(500)
    n, m = 5, 7

    a = make_stable_continuous_complex(n, seed=500)
    a_orig = a.copy()

    b = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    b = np.asfortranarray(b)
    b_orig = b.copy()

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    scale = ctypes.c_double(0.0)
    w = np.zeros(n, dtype=np.complex128)
    ldwork = max(1, n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    lzwork = max(1, 2*n + max(min(n, m) - 2, 0))
    zwork = np.zeros(lzwork, dtype=np.complex128)
    info = ctypes.c_int(0)

    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:m, :n] = b

    sb03oz(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lzwork,
        ctypes.byref(info)
    )

    assert info.value == 0
    assert 0 < scale.value <= 1.0

    u = np.triu(b_padded[:n, :n])
    x = u.conj().T @ u

    rhs = -scale.value**2 * b_orig.conj().T @ b_orig
    residual = a_orig.conj().T @ x + x @ a_orig - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-8)


def test_positive_semidefinite(sb03oz):
    """
    Validate X = U^H * U is positive semi-definite (Hermitian positive).

    Random seed: 600 (for reproducibility)
    """
    np.random.seed(600)
    n, m = 4, 5

    a = make_stable_continuous_complex(n, seed=600)

    b = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    b = np.asfortranarray(b)

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    scale = ctypes.c_double(0.0)
    w = np.zeros(n, dtype=np.complex128)
    ldwork = max(1, n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    lzwork = max(1, 2*n + max(min(n, m) - 2, 0))
    zwork = np.zeros(lzwork, dtype=np.complex128)
    info = ctypes.c_int(0)

    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:m, :n] = b

    sb03oz(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lzwork,
        ctypes.byref(info)
    )

    if info.value == 0:
        u = np.triu(b_padded[:n, :n])
        x = u.conj().T @ u
        eig = np.linalg.eigvalsh(x)
        assert all(e >= -1e-10 for e in eig), "X not positive semi-definite"


def test_upper_triangular_output(sb03oz):
    """
    Validate output U remains upper triangular with real non-negative diagonal.

    Random seed: 700 (for reproducibility)
    """
    np.random.seed(700)
    n, m = 4, 5

    a = make_stable_continuous_complex(n, seed=700)

    b = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    b = np.asfortranarray(b)

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    scale = ctypes.c_double(0.0)
    w = np.zeros(n, dtype=np.complex128)
    ldwork = max(1, n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    lzwork = max(1, 2*n + max(min(n, m) - 2, 0))
    zwork = np.zeros(lzwork, dtype=np.complex128)
    info = ctypes.c_int(0)

    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:m, :n] = b

    sb03oz(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lzwork,
        ctypes.byref(info)
    )

    if info.value == 0:
        u = b_padded[:n, :n]
        assert np.allclose(np.tril(u, -1), 0, atol=1e-14), "U not upper triangular"
        for i in range(n):
            diag_val = u[i, i]
            assert abs(diag_val.imag) < 1e-14, f"Diagonal {i} not real"
            assert diag_val.real >= -1e-14, f"Diagonal {i} negative"


def test_eigenvalue_check(sb03oz):
    """
    Validate eigenvalues W returned match eigenvalues of A.

    Random seed: 800 (for reproducibility)
    """
    np.random.seed(800)
    n, m = 4, 5

    a = make_stable_continuous_complex(n, seed=800)
    a_orig = a.copy()

    b = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    b = np.asfortranarray(b)

    q = np.zeros((n, n), order='F', dtype=np.complex128)

    scale = ctypes.c_double(0.0)
    w = np.zeros(n, dtype=np.complex128)
    ldwork = max(1, n)
    dwork = np.zeros(ldwork, dtype=np.float64)
    lzwork = max(1, 2*n + max(min(n, m) - 2, 0))
    zwork = np.zeros(lzwork, dtype=np.complex128)
    info = ctypes.c_int(0)

    ldb = max(n, m)
    b_padded = np.zeros((ldb, max(m, n)), order='F', dtype=np.complex128)
    b_padded[:m, :n] = b

    sb03oz(
        b"C", b"N", b"N", n, m,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b_padded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        ctypes.byref(scale),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lzwork,
        ctypes.byref(info)
    )

    if info.value == 0:
        expected_eig = np.linalg.eigvals(a_orig)
        np.testing.assert_allclose(sorted(w.real), sorted(expected_eig.real), rtol=1e-10)
        np.testing.assert_allclose(sorted(w.imag), sorted(expected_eig.imag), rtol=1e-10)
