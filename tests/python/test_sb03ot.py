"""
Tests for SB03OT - Reduced Lyapunov equation solver for triangular factors.

SB03OT solves for X = op(U)'*op(U) either the stable continuous-time Lyapunov equation:
    op(S)'*X + X*op(S) = -scale^2*op(R)'*op(R)
or the convergent discrete-time Lyapunov equation:
    op(S)'*X*op(S) - X = -scale^2*op(R)'*op(R)

where S is block upper triangular (real Schur form), R is upper triangular.
The output U is upper triangular and overwrites R.

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
def sb03ot(lib):
    """Get the sb03ot function with proper signature."""
    func = lib.sb03ot
    func.argtypes = [
        ctypes.c_bool,                    # discr
        ctypes.c_bool,                    # ltrans
        ctypes.c_int,                     # n
        ctypes.POINTER(ctypes.c_double),  # s
        ctypes.c_int,                     # lds
        ctypes.POINTER(ctypes.c_double),  # r
        ctypes.c_int,                     # ldr
        ctypes.POINTER(ctypes.c_double),  # scale
        ctypes.POINTER(ctypes.c_double),  # dwork
        ctypes.POINTER(ctypes.c_int),     # info
    ]
    func.restype = None
    return func


def make_stable_continuous_schur(n, seed=42):
    """
    Create an n-by-n block upper triangular matrix in real Schur form
    with stable eigenvalues (negative real parts).
    
    Random seed: {seed} (for reproducibility)
    """
    np.random.seed(seed)
    s = np.zeros((n, n), order='F', dtype=np.float64)
    
    i = 0
    while i < n:
        if i == n - 1 or np.random.rand() < 0.5:
            s[i, i] = -0.5 - np.random.rand()
            i += 1
        else:
            alpha = -0.5 - np.random.rand()
            omega = 0.1 + np.random.rand()
            s[i, i] = alpha
            s[i+1, i+1] = alpha
            s[i, i+1] = omega
            s[i+1, i] = -omega
            i += 2
    
    for i in range(n):
        for j in range(i+2, n):
            s[i, j] = 0.5 * np.random.randn()
    
    return s


def make_stable_discrete_schur(n, seed=42):
    """
    Create an n-by-n block upper triangular matrix in real Schur form
    with convergent eigenvalues (inside unit circle).
    
    Random seed: {seed} (for reproducibility)
    """
    np.random.seed(seed)
    s = np.zeros((n, n), order='F', dtype=np.float64)
    
    i = 0
    while i < n:
        if i == n - 1 or np.random.rand() < 0.5:
            s[i, i] = (np.random.rand() - 0.5) * 1.6
            i += 1
        else:
            r = 0.3 + np.random.rand() * 0.5
            theta = np.random.rand() * np.pi
            c, si = np.cos(theta), np.sin(theta)
            s[i, i] = r * c
            s[i+1, i+1] = r * c
            s[i, i+1] = r * si
            s[i+1, i] = -r * si
            i += 2
    
    for i in range(n):
        for j in range(i+2, n):
            s[i, j] = 0.3 * np.random.randn()
    
    return s


def make_upper_triangular(n, seed=42):
    """
    Create an n-by-n upper triangular matrix.
    
    Random seed: {seed} (for reproducibility)
    """
    np.random.seed(seed)
    r = np.triu(np.random.randn(n, n))
    return np.asfortranarray(r)


"""Test continuous-time Lyapunov equation op(K)=K."""

def test_1x1_basic(sb03ot):
    """
    Test 1x1 continuous-time case.
    
    Equation: s*x + x*s = -scale^2 * r^2, where x = u^2.
    Solution: u = r / sqrt(-2*s).
    """
    n = 1
    s = np.array([[-2.0]], order='F', dtype=np.float64)
    r = np.array([[1.0]], order='F', dtype=np.float64)
    r_orig = r.copy()
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        False, False, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = r
    x = u.T @ u
    
    rhs = -scale.value**2 * r_orig.T @ r_orig
    residual = s.T @ x + x @ s - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)

def test_2x2_single_blocks(sb03ot):
    """
    Test 2x2 continuous-time with two 1x1 blocks (diagonal S).
    
    Random seed: 100 (for reproducibility)
    """
    n = 2
    s = np.array([[-1.0, 0.5], [0.0, -2.0]], order='F', dtype=np.float64)
    r = np.array([[1.0, 0.3], [0.0, 0.8]], order='F', dtype=np.float64)
    r_orig = r.copy()
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        False, False, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = r
    x = u.T @ u
    
    rhs = -scale.value**2 * r_orig.T @ r_orig
    residual = s.T @ x + x @ s - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)

def test_2x2_complex_block(sb03ot):
    """
    Test 2x2 continuous-time with 2x2 block (complex conjugate eigenvalues).
    
    S has eigenvalues -1 +/- 2i.
    """
    n = 2
    s = np.array([[-1.0, 2.0], [-2.0, -1.0]], order='F', dtype=np.float64)
    r = np.array([[1.0, 0.5], [0.0, 1.0]], order='F', dtype=np.float64)
    r_orig = r.copy()
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        False, False, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = r
    x = u.T @ u
    
    rhs = -scale.value**2 * r_orig.T @ r_orig
    residual = s.T @ x + x @ s - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)

def test_4x4_mixed_blocks(sb03ot):
    """
    Test 4x4 continuous-time with mixed 1x1 and 2x2 blocks.
    
    S has structure: 2x2 block, then two 1x1 blocks.
    Random seed: 200 (for reproducibility)
    """
    np.random.seed(200)
    n = 4
    s = np.zeros((n, n), order='F', dtype=np.float64)
    s[0, 0] = -1.0; s[0, 1] = 1.5; s[1, 0] = -1.5; s[1, 1] = -1.0
    s[2, 2] = -2.0
    s[3, 3] = -3.0
    s[0, 2] = 0.3; s[0, 3] = 0.2
    s[1, 2] = 0.1; s[1, 3] = 0.4
    s[2, 3] = 0.5
    
    r = make_upper_triangular(n, seed=201)
    r_orig = r.copy()
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        False, False, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(r)
    x = u.T @ u
    
    rhs = -scale.value**2 * r_orig.T @ r_orig
    residual = s.T @ x + x @ s - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


"""Test continuous-time Lyapunov equation op(K)=K'."""

def test_2x2_transpose(sb03ot):
    """
    Test 2x2 continuous-time with transpose.
    
    Equation: S*X + X*S' = -scale^2 * R*R', where X = U*U'.
    """
    n = 2
    s = np.array([[-1.0, 0.5], [0.0, -2.0]], order='F', dtype=np.float64)
    r = np.array([[1.0, 0.3], [0.0, 0.8]], order='F', dtype=np.float64)
    r_orig = r.copy()
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        False, True, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(r)
    x = u @ u.T
    
    rhs = -scale.value**2 * r_orig @ r_orig.T
    residual = s @ x + x @ s.T - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)

def test_4x4_transpose(sb03ot):
    """
    Test 4x4 continuous-time with transpose.
    
    Random seed: 300 (for reproducibility)
    """
    n = 4
    s = make_stable_continuous_schur(n, seed=300)
    r = make_upper_triangular(n, seed=301)
    r_orig = r.copy()
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        False, True, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(r)
    x = u @ u.T
    
    rhs = -scale.value**2 * r_orig @ r_orig.T
    residual = s @ x + x @ s.T - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


"""Test discrete-time Lyapunov equation op(K)=K."""

def test_1x1_discrete(sb03ot):
    """
    Test 1x1 discrete-time case.
    
    Equation: s*x*s - x = -scale^2 * r^2, where x = u^2.
    """
    n = 1
    s = np.array([[0.5]], order='F', dtype=np.float64)
    r = np.array([[1.0]], order='F', dtype=np.float64)
    r_orig = r.copy()
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        True, False, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = r
    x = u.T @ u
    
    rhs = -scale.value**2 * r_orig.T @ r_orig
    residual = s.T @ x @ s - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)

def test_2x2_discrete_complex_block(sb03ot):
    """
    Test 2x2 discrete-time with 2x2 block (complex conjugate eigenvalues).
    
    S has eigenvalues 0.6*exp(+/-i*pi/3), modulus = 0.6 < 1.
    """
    n = 2
    r_mod = 0.6
    theta = np.pi/3
    c, si = np.cos(theta), np.sin(theta)
    s = np.array([[r_mod*c, r_mod*si], [-r_mod*si, r_mod*c]], order='F', dtype=np.float64)
    r = np.array([[1.0, 0.4], [0.0, 0.9]], order='F', dtype=np.float64)
    r_orig = r.copy()
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        True, False, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(r)
    x = u.T @ u
    
    rhs = -scale.value**2 * r_orig.T @ r_orig
    residual = s.T @ x @ s - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)

def test_4x4_discrete_mixed(sb03ot):
    """
    Test 4x4 discrete-time with mixed blocks.
    
    Random seed: 400 (for reproducibility)
    """
    n = 4
    s = make_stable_discrete_schur(n, seed=400)
    r = make_upper_triangular(n, seed=401)
    r_orig = r.copy()
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        True, False, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(r)
    x = u.T @ u
    
    rhs = -scale.value**2 * r_orig.T @ r_orig
    residual = s.T @ x @ s - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


"""Test discrete-time Lyapunov equation op(K)=K'."""

def test_2x2_discrete_transpose(sb03ot):
    """
    Test 2x2 discrete-time with transpose.
    
    Equation: S*X*S' - X = -scale^2 * R*R', where X = U*U'.
    """
    n = 2
    s = np.array([[0.5, 0.3], [0.0, 0.6]], order='F', dtype=np.float64)
    r = np.array([[1.0, 0.2], [0.0, 0.7]], order='F', dtype=np.float64)
    r_orig = r.copy()
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        True, True, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(r)
    x = u @ u.T
    
    rhs = -scale.value**2 * r_orig @ r_orig.T
    residual = s @ x @ s.T - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)

def test_4x4_discrete_transpose(sb03ot):
    """
    Test 4x4 discrete-time with transpose.
    
    Random seed: 500 (for reproducibility)
    """
    n = 4
    s = make_stable_discrete_schur(n, seed=500)
    r = make_upper_triangular(n, seed=501)
    r_orig = r.copy()
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        True, True, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(r)
    x = u @ u.T
    
    rhs = -scale.value**2 * r_orig @ r_orig.T
    residual = s @ x @ s.T - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


"""Test edge cases and error handling."""

def test_n_zero(sb03ot):
    """Test n=0 returns immediately with success."""
    n = 0
    s = np.zeros((1, 1), order='F', dtype=np.float64)
    r = np.zeros((1, 1), order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4, dtype=np.float64)
    
    sb03ot(
        False, False, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert scale.value == 1.0

def test_unstable_continuous_returns_info2(sb03ot):
    """
    Test that unstable S (positive real eigenvalue) returns info=2.
    """
    n = 1
    s = np.array([[1.0]], order='F', dtype=np.float64)
    r = np.array([[1.0]], order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        False, False, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 2

def test_non_convergent_discrete_returns_info2(sb03ot):
    """
    Test that non-convergent S (eigenvalue outside unit circle) returns info=2.
    """
    n = 1
    s = np.array([[1.5]], order='F', dtype=np.float64)
    r = np.array([[1.0]], order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        True, False, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 2

def test_block_larger_than_2x2_returns_info3(sb03ot):
    """
    Test that consecutive non-zero subdiagonals (implying >2x2 block) returns info=3.
    """
    n = 3
    s = np.array([
        [1.0, 0.5, 0.2],
        [1.0, 1.0, 0.3],
        [0.0, 1.0, 1.0]
    ], order='F', dtype=np.float64)
    r = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        False, False, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 3

def test_2x2_block_real_eigenvalues_returns_info4(sb03ot):
    """
    Test that 2x2 block with real eigenvalues returns info=4.
    """
    n = 2
    s = np.array([[1.0, 2.0], [0.5, 1.0]], order='F', dtype=np.float64)
    r = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        False, False, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 4


"""Test larger systems for robustness."""

def test_6x6_continuous(sb03ot):
    """
    Test 6x6 continuous-time.
    
    Random seed: 600 (for reproducibility)
    """
    n = 6
    s = make_stable_continuous_schur(n, seed=600)
    r = make_upper_triangular(n, seed=601)
    r_orig = r.copy()
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        False, False, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(r)
    x = u.T @ u
    
    rhs = -scale.value**2 * r_orig.T @ r_orig
    residual = s.T @ x + x @ s - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-8)

def test_8x8_discrete_transpose(sb03ot):
    """
    Test 8x8 discrete-time with transpose.
    
    Random seed: 800 (for reproducibility)
    """
    n = 8
    s = make_stable_discrete_schur(n, seed=800)
    r = make_upper_triangular(n, seed=801)
    r_orig = r.copy()
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4*n, dtype=np.float64)
    
    sb03ot(
        True, True, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == 0
    assert 0 < scale.value <= 1.0
    
    u = np.triu(r)
    x = u @ u.T
    
    rhs = -scale.value**2 * r_orig @ r_orig.T
    residual = s @ x @ s.T - x - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-8)


"""Test invalid input parameter handling."""

def test_negative_n_returns_info_minus3(sb03ot):
    """Test that n < 0 returns info = -3."""
    n = -1
    s = np.zeros((1, 1), order='F', dtype=np.float64)
    r = np.zeros((1, 1), order='F', dtype=np.float64)
    
    scale = ctypes.c_double(0.0)
    info = ctypes.c_int(0)
    dwork = np.zeros(4, dtype=np.float64)
    
    sb03ot(
        False, False, n,
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(scale),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(info)
    )
    
    assert info.value == -3
