"""
Tests for SB04PX - 2x2 Sylvester equation solver.

SB04PX solves for X in: op(TL)*X*op(TR) + ISGN*X = SCALE*B
where TL is N1-by-N1, TR is N2-by-N2, B is N1-by-N2, and ISGN = 1 or -1.

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
def sb04px(lib):
    """Get the sb04px function with proper signature."""
    func = lib.sb04px
    func.argtypes = [
        ctypes.c_bool,               # ltranl
        ctypes.c_bool,               # ltranr
        ctypes.c_int,                # isgn
        ctypes.c_int,                # n1
        ctypes.c_int,                # n2
        ctypes.POINTER(ctypes.c_double),  # tl
        ctypes.c_int,                # ldtl
        ctypes.POINTER(ctypes.c_double),  # tr
        ctypes.c_int,                # ldtr
        ctypes.POINTER(ctypes.c_double),  # b
        ctypes.c_int,                # ldb
        ctypes.POINTER(ctypes.c_double),  # scale
        ctypes.POINTER(ctypes.c_double),  # x
        ctypes.c_int,                # ldx
        ctypes.POINTER(ctypes.c_double),  # xnorm
        ctypes.POINTER(ctypes.c_int),     # info
    ]
    func.restype = None
    return func


def test_sb04px_1x1_basic(sb04px):
    """
    Test 1x1 case: TL*X*TR + ISGN*X = SCALE*B

    For TL=2, TR=3, ISGN=1, B=12:
    2*X*3 + X = 12 => 7*X = 12 => X = 12/7 ≈ 1.714286

    Random seed: N/A (deterministic test data)
    """
    tl = np.array([[2.0]], order='F', dtype=np.float64)
    tr = np.array([[3.0]], order='F', dtype=np.float64)
    b = np.array([[12.0]], order='F', dtype=np.float64)
    x = np.zeros((1, 1), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    xnorm = ctypes.c_double(0.0)
    info = ctypes.c_int(0)

    sb04px(
        False, False, 1, 1, 1,
        tl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        tr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(scale),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(xnorm),
        ctypes.byref(info)
    )

    assert info.value == 0
    assert scale.value == 1.0
    expected_x = 12.0 / 7.0
    np.testing.assert_allclose(x[0, 0], expected_x, rtol=1e-14)
    np.testing.assert_allclose(xnorm.value, abs(expected_x), rtol=1e-14)


def test_sb04px_1x1_with_transpose(sb04px):
    """
    Test 1x1 case with transpose flags (no effect for scalars).

    TL=4, TR=2, ISGN=-1, B=6:
    4*X*2 - X = 6 => 7*X = 6 => X = 6/7

    Random seed: N/A (deterministic test data)
    """
    tl = np.array([[4.0]], order='F', dtype=np.float64)
    tr = np.array([[2.0]], order='F', dtype=np.float64)
    b = np.array([[6.0]], order='F', dtype=np.float64)
    x = np.zeros((1, 1), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    xnorm = ctypes.c_double(0.0)
    info = ctypes.c_int(0)

    sb04px(
        True, True, -1, 1, 1,
        tl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        tr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(scale),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(xnorm),
        ctypes.byref(info)
    )

    assert info.value == 0
    assert scale.value == 1.0
    expected_x = 6.0 / 7.0
    np.testing.assert_allclose(x[0, 0], expected_x, rtol=1e-14)


def test_sb04px_2x2_basic(sb04px):
    """
    Test 2x2 case: TL*X*TR + X = SCALE*B (no transpose, ISGN=1)

    Uses simple matrices to verify the 4x4 linear system solution.
    TL = [[1, 0], [0, 2]], TR = [[1, 0], [0, 1]]
    This simplifies to:
    X[0,0] + X[0,0] = B[0,0] => 2*X[0,0] = B[0,0]
    2*X[1,0] + X[1,0] = B[1,0] => 3*X[1,0] = B[1,0]
    etc.

    Random seed: N/A (deterministic test data)
    """
    tl = np.array([[1.0, 0.0], [0.0, 2.0]], order='F', dtype=np.float64)
    tr = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=np.float64)
    b = np.array([[4.0, 6.0], [9.0, 15.0]], order='F', dtype=np.float64)
    x = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    xnorm = ctypes.c_double(0.0)
    info = ctypes.c_int(0)

    sb04px(
        False, False, 1, 2, 2,
        tl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        tr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(xnorm),
        ctypes.byref(info)
    )

    assert info.value == 0
    assert scale.value == 1.0

    expected_x = np.array([[2.0, 3.0], [3.0, 5.0]], order='F', dtype=np.float64)
    np.testing.assert_allclose(x, expected_x, rtol=1e-14)


def test_sb04px_1x2_case(sb04px):
    """
    Test 1x2 case: TL*[X11 X12]*op(TR) + [X11 X12] = B

    TL = [[2]], TR = [[1, 0.5], [0.5, 1]], B = [[6, 5]]

    Random seed: N/A (deterministic test data)
    """
    tl = np.array([[2.0]], order='F', dtype=np.float64)
    tr = np.array([[1.0, 0.5], [0.5, 1.0]], order='F', dtype=np.float64)
    b = np.array([[6.0, 5.0]], order='F', dtype=np.float64)
    x = np.zeros((1, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    xnorm = ctypes.c_double(0.0)
    info = ctypes.c_int(0)

    sb04px(
        False, False, 1, 1, 2,
        tl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        tr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(scale),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(xnorm),
        ctypes.byref(info)
    )

    assert info.value == 0
    assert scale.value == 1.0

    tl_op = tl
    tr_op = tr
    residual = tl_op @ x @ tr_op + x - scale.value * b
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)


def test_sb04px_2x1_case(sb04px):
    """
    Test 2x1 case: op(TL)*[X11; X21]*TR + [X11; X21] = B

    TL = [[1, 0.3], [0.3, 2]], TR = [[3]], B = [[8], [14]]

    Random seed: N/A (deterministic test data)
    """
    tl = np.array([[1.0, 0.3], [0.3, 2.0]], order='F', dtype=np.float64)
    tr = np.array([[3.0]], order='F', dtype=np.float64)
    b = np.array([[8.0], [14.0]], order='F', dtype=np.float64)
    x = np.zeros((2, 1), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    xnorm = ctypes.c_double(0.0)
    info = ctypes.c_int(0)

    sb04px(
        False, False, 1, 2, 1,
        tl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        tr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(xnorm),
        ctypes.byref(info)
    )

    assert info.value == 0
    assert scale.value == 1.0

    tl_op = tl
    tr_op = tr
    residual = tl_op @ x @ tr_op + x - scale.value * b
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)


def test_sb04px_n1_zero(sb04px):
    """
    Test edge case: N1=0 (empty problem).

    Should return immediately with scale=1, xnorm=0.
    """
    tl = np.array([[1.0]], order='F', dtype=np.float64)
    tr = np.array([[1.0]], order='F', dtype=np.float64)
    b = np.array([[1.0]], order='F', dtype=np.float64)
    x = np.zeros((1, 1), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    xnorm = ctypes.c_double(-1.0)
    info = ctypes.c_int(-999)

    sb04px(
        False, False, 1, 0, 1,
        tl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        tr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(scale),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(xnorm),
        ctypes.byref(info)
    )

    assert info.value == 0
    assert xnorm.value == 0.0


def test_sb04px_n2_zero(sb04px):
    """
    Test edge case: N2=0 (empty problem).

    Should return immediately with scale=1, xnorm=0.
    """
    tl = np.array([[1.0]], order='F', dtype=np.float64)
    tr = np.array([[1.0]], order='F', dtype=np.float64)
    b = np.array([[1.0]], order='F', dtype=np.float64)
    x = np.zeros((1, 1), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    xnorm = ctypes.c_double(-1.0)
    info = ctypes.c_int(-999)

    sb04px(
        False, False, 1, 1, 0,
        tl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        tr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(scale),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        ctypes.byref(xnorm),
        ctypes.byref(info)
    )

    assert info.value == 0
    assert xnorm.value == 0.0


def test_sb04px_2x2_transpose_left(sb04px):
    """
    Test 2x2 case with LTRANL=True: TL'*X*TR + X = B

    Random seed: N/A (deterministic test data)
    """
    tl = np.array([[1.0, 0.2], [0.3, 2.0]], order='F', dtype=np.float64)
    tr = np.array([[1.5, 0.1], [0.1, 0.5]], order='F', dtype=np.float64)
    b = np.array([[5.0, 3.0], [8.0, 6.0]], order='F', dtype=np.float64)
    x = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    xnorm = ctypes.c_double(0.0)
    info = ctypes.c_int(0)

    sb04px(
        True, False, 1, 2, 2,
        tl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        tr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(xnorm),
        ctypes.byref(info)
    )

    assert info.value == 0
    assert scale.value == 1.0

    tl_op = tl.T
    tr_op = tr
    residual = tl_op @ x @ tr_op + x - scale.value * b
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)


def test_sb04px_2x2_transpose_right(sb04px):
    """
    Test 2x2 case with LTRANR=True: TL*X*TR' + X = B

    Random seed: N/A (deterministic test data)
    """
    tl = np.array([[1.0, 0.2], [0.3, 2.0]], order='F', dtype=np.float64)
    tr = np.array([[1.5, 0.1], [0.1, 0.5]], order='F', dtype=np.float64)
    b = np.array([[5.0, 3.0], [8.0, 6.0]], order='F', dtype=np.float64)
    x = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    xnorm = ctypes.c_double(0.0)
    info = ctypes.c_int(0)

    sb04px(
        False, True, 1, 2, 2,
        tl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        tr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(xnorm),
        ctypes.byref(info)
    )

    assert info.value == 0
    assert scale.value == 1.0

    tl_op = tl
    tr_op = tr.T
    residual = tl_op @ x @ tr_op + x - scale.value * b
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)


def test_sb04px_2x2_both_transpose(sb04px):
    """
    Test 2x2 case with both transposed: TL'*X*TR' + X = B

    Random seed: N/A (deterministic test data)
    """
    tl = np.array([[1.0, 0.2], [0.3, 2.0]], order='F', dtype=np.float64)
    tr = np.array([[1.5, 0.1], [0.1, 0.5]], order='F', dtype=np.float64)
    b = np.array([[5.0, 3.0], [8.0, 6.0]], order='F', dtype=np.float64)
    x = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    xnorm = ctypes.c_double(0.0)
    info = ctypes.c_int(0)

    sb04px(
        True, True, 1, 2, 2,
        tl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        tr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(xnorm),
        ctypes.byref(info)
    )

    assert info.value == 0
    assert scale.value == 1.0

    tl_op = tl.T
    tr_op = tr.T
    residual = tl_op @ x @ tr_op + x - scale.value * b
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)


def test_sb04px_isgn_minus_one(sb04px):
    """
    Test with ISGN=-1: TL*X*TR - X = B

    Random seed: N/A (deterministic test data)
    """
    tl = np.array([[2.0, 0.1], [0.1, 3.0]], order='F', dtype=np.float64)
    tr = np.array([[1.0, 0.2], [0.2, 1.5]], order='F', dtype=np.float64)
    b = np.array([[4.0, 2.0], [6.0, 8.0]], order='F', dtype=np.float64)
    x = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    xnorm = ctypes.c_double(0.0)
    info = ctypes.c_int(0)

    sb04px(
        False, False, -1, 2, 2,
        tl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        tr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(xnorm),
        ctypes.byref(info)
    )

    assert info.value == 0
    assert scale.value == 1.0

    residual = tl @ x @ tr - x - scale.value * b
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)


def test_sb04px_residual_property(sb04px):
    """
    Property test: verify equation residual is zero for random matrices.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    tl = np.random.randn(2, 2).astype(np.float64, order='F')
    tr = np.random.randn(2, 2).astype(np.float64, order='F')
    b = np.random.randn(2, 2).astype(np.float64, order='F')
    x = np.zeros((2, 2), order='F', dtype=np.float64)
    scale = ctypes.c_double(0.0)
    xnorm = ctypes.c_double(0.0)
    info = ctypes.c_int(0)

    sb04px(
        False, False, 1, 2, 2,
        tl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        tr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(scale),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
        ctypes.byref(xnorm),
        ctypes.byref(info)
    )

    residual = tl @ x @ tr + x - scale.value * b
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)
