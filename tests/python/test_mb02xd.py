"""
Tests for MB02XD - Solve A'*A*X = B using Cholesky factorization.

MB02XD computes the solution to a system of linear equations A'*A*X = B,
where A'*A is a symmetric positive definite matrix, using Cholesky
factorization.

Since MB02XD is a low-level routine without Python wrapper, we test the
underlying C implementation via ctypes.
"""

import ctypes
import numpy as np
import os
import pytest


def get_slicot_lib():
    """Load the SLICOT shared library."""
    import glob
    import sys
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    if sys.platform == 'darwin':
        build_dirs = [
            os.path.join(base_dir, "build/macos-arm64-debug/src"),
            os.path.join(base_dir, "build/macos-arm64-release/src"),
        ]
    else:
        build_dirs = [
            os.path.join(base_dir, "build/linux-x64-debug/src"),
            os.path.join(base_dir, "build/linux-x64-release/src"),
            os.path.join(base_dir, "build/linux-x64-debug-sanitizers/src"),
        ]
    for bd in build_dirs:
        if os.path.exists(bd):
            libs = glob.glob(os.path.join(bd, 'libslicot.*'))
            if libs:
                return ctypes.CDLL(libs[0])
    pytest.skip("SLICOT library not found")


def test_mb02xd_basic_full_upper():
    """
    Test MB02XD with full storage, upper triangle.

    Solve A'*A*X = B where A is a 4x3 matrix (M=4, N=3).

    Uses FORM='S' (standard form), STOR='F' (full), UPLO='U' (upper).
    """
    lib = get_slicot_lib()

    m, n, nrhs = 4, 3, 2

    np.random.seed(42)
    A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 10.0],
        [1.0, 1.0, 2.0]
    ], order='F', dtype=np.float64)

    B = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ], order='F', dtype=np.float64)

    AtA = A.T @ A
    X_expected = np.linalg.solve(AtA, B)

    lda = m
    ldb = n
    ldata = n
    ata = np.zeros((n, n), order='F', dtype=np.float64)
    dwork = np.zeros(1, dtype=np.float64)
    ldwork = 1

    info = ctypes.c_int(0)

    lib.mb02xd.argtypes = [
        ctypes.c_char_p,  # form
        ctypes.c_char_p,  # stor
        ctypes.c_char_p,  # uplo
        ctypes.c_void_p,  # f (callback, NULL for FORM='S')
        ctypes.POINTER(ctypes.c_int),  # m
        ctypes.POINTER(ctypes.c_int),  # n
        ctypes.POINTER(ctypes.c_int),  # nrhs
        ctypes.c_void_p,  # ipar (not used for FORM='S')
        ctypes.POINTER(ctypes.c_int),  # lipar
        ctypes.c_void_p,  # dpar (not used for FORM='S')
        ctypes.POINTER(ctypes.c_int),  # ldpar
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),  # a
        ctypes.POINTER(ctypes.c_int),  # lda
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),  # b
        ctypes.POINTER(ctypes.c_int),  # ldb
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),  # ata
        ctypes.POINTER(ctypes.c_int),  # ldata
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # dwork
        ctypes.POINTER(ctypes.c_int),  # ldwork
        ctypes.POINTER(ctypes.c_int),  # info
    ]
    lib.mb02xd.restype = None

    m_c = ctypes.c_int(m)
    n_c = ctypes.c_int(n)
    nrhs_c = ctypes.c_int(nrhs)
    lipar_c = ctypes.c_int(0)
    ldpar_c = ctypes.c_int(0)
    lda_c = ctypes.c_int(lda)
    ldb_c = ctypes.c_int(ldb)
    ldata_c = ctypes.c_int(ldata)
    ldwork_c = ctypes.c_int(ldwork)

    lib.mb02xd(
        b"S",  # form - standard
        b"F",  # stor - full
        b"U",  # uplo - upper
        None,  # f callback (not used)
        ctypes.byref(m_c),
        ctypes.byref(n_c),
        ctypes.byref(nrhs_c),
        None,  # ipar
        ctypes.byref(lipar_c),
        None,  # dpar
        ctypes.byref(ldpar_c),
        A,
        ctypes.byref(lda_c),
        B,
        ctypes.byref(ldb_c),
        ata,
        ctypes.byref(ldata_c),
        dwork,
        ctypes.byref(ldwork_c),
        ctypes.byref(info)
    )

    assert info.value == 0, f"MB02XD failed with info={info.value}"
    np.testing.assert_allclose(B, X_expected, rtol=1e-13, atol=1e-14)

    L_expected = np.linalg.cholesky(AtA).T
    np.testing.assert_allclose(np.triu(ata), L_expected, rtol=1e-13, atol=1e-14)


def test_mb02xd_full_lower():
    """
    Test MB02XD with full storage, lower triangle.

    Random seed: 123 (for reproducibility)
    """
    lib = get_slicot_lib()

    m, n, nrhs = 5, 3, 1

    np.random.seed(123)
    A = np.random.randn(m, n).astype(np.float64, order='F')
    B = np.random.randn(n, nrhs).astype(np.float64, order='F')

    AtA = A.T @ A
    X_expected = np.linalg.solve(AtA, B)

    lda = m
    ldb = n
    ldata = n
    ata = np.zeros((n, n), order='F', dtype=np.float64)
    dwork = np.zeros(1, dtype=np.float64)

    info = ctypes.c_int(0)

    lib.mb02xd.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.mb02xd.restype = None

    lib.mb02xd(
        b"S", b"F", b"L", None,
        ctypes.byref(ctypes.c_int(m)),
        ctypes.byref(ctypes.c_int(n)),
        ctypes.byref(ctypes.c_int(nrhs)),
        None, ctypes.byref(ctypes.c_int(0)),
        None, ctypes.byref(ctypes.c_int(0)),
        A, ctypes.byref(ctypes.c_int(lda)),
        B, ctypes.byref(ctypes.c_int(ldb)),
        ata, ctypes.byref(ctypes.c_int(ldata)),
        dwork, ctypes.byref(ctypes.c_int(1)),
        ctypes.byref(info)
    )

    assert info.value == 0, f"MB02XD failed with info={info.value}"
    np.testing.assert_allclose(B, X_expected, rtol=1e-13, atol=1e-14)


def test_mb02xd_packed_upper():
    """
    Test MB02XD with packed storage, upper triangle.

    For packed storage with UPLO='U', the upper triangle is stored
    column by column: A(1,1), A(1,2), A(2,2), A(1,3), A(2,3), A(3,3), ...

    Random seed: 456 (for reproducibility)
    """
    lib = get_slicot_lib()

    m, n, nrhs = 4, 3, 2

    np.random.seed(456)
    A = np.random.randn(m, n).astype(np.float64, order='F')
    B = np.random.randn(n, nrhs).astype(np.float64, order='F')

    AtA = A.T @ A
    X_expected = np.linalg.solve(AtA, B)

    lda = m
    ldb = n
    packed_size = n * (n + 1) // 2
    ata = np.zeros(packed_size, dtype=np.float64)
    dwork = np.zeros(1, dtype=np.float64)

    info = ctypes.c_int(0)

    lib.mb02xd.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.mb02xd.restype = None

    lib.mb02xd(
        b"S", b"P", b"U", None,
        ctypes.byref(ctypes.c_int(m)),
        ctypes.byref(ctypes.c_int(n)),
        ctypes.byref(ctypes.c_int(nrhs)),
        None, ctypes.byref(ctypes.c_int(0)),
        None, ctypes.byref(ctypes.c_int(0)),
        A, ctypes.byref(ctypes.c_int(lda)),
        B, ctypes.byref(ctypes.c_int(ldb)),
        ata, ctypes.byref(ctypes.c_int(1)),
        dwork, ctypes.byref(ctypes.c_int(1)),
        ctypes.byref(info)
    )

    assert info.value == 0, f"MB02XD failed with info={info.value}"
    np.testing.assert_allclose(B, X_expected, rtol=1e-13, atol=1e-14)


def test_mb02xd_packed_lower():
    """
    Test MB02XD with packed storage, lower triangle.

    For packed storage with UPLO='L', the lower triangle is stored
    column by column: A(1,1), A(2,1), A(3,1), ..., A(2,2), A(3,2), ...

    Random seed: 789 (for reproducibility)
    """
    lib = get_slicot_lib()

    m, n, nrhs = 6, 4, 3

    np.random.seed(789)
    A = np.random.randn(m, n).astype(np.float64, order='F')
    B = np.random.randn(n, nrhs).astype(np.float64, order='F')

    AtA = A.T @ A
    X_expected = np.linalg.solve(AtA, B)

    lda = m
    ldb = n
    packed_size = n * (n + 1) // 2
    ata = np.zeros(packed_size, dtype=np.float64)
    dwork = np.zeros(1, dtype=np.float64)

    info = ctypes.c_int(0)

    lib.mb02xd.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.mb02xd.restype = None

    lib.mb02xd(
        b"S", b"P", b"L", None,
        ctypes.byref(ctypes.c_int(m)),
        ctypes.byref(ctypes.c_int(n)),
        ctypes.byref(ctypes.c_int(nrhs)),
        None, ctypes.byref(ctypes.c_int(0)),
        None, ctypes.byref(ctypes.c_int(0)),
        A, ctypes.byref(ctypes.c_int(lda)),
        B, ctypes.byref(ctypes.c_int(ldb)),
        ata, ctypes.byref(ctypes.c_int(1)),
        dwork, ctypes.byref(ctypes.c_int(1)),
        ctypes.byref(info)
    )

    assert info.value == 0, f"MB02XD failed with info={info.value}"
    np.testing.assert_allclose(B, X_expected, rtol=1e-13, atol=1e-14)


def test_mb02xd_singular_matrix():
    """
    Test MB02XD with singular A'*A matrix (rank deficient A).

    Should return info > 0 indicating singularity.
    """
    lib = get_slicot_lib()

    m, n, nrhs = 4, 3, 1

    A = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0]
    ], order='F', dtype=np.float64)

    B = np.array([[1.0], [2.0], [3.0]], order='F', dtype=np.float64)

    lda = m
    ldb = n
    ldata = n
    ata = np.zeros((n, n), order='F', dtype=np.float64)
    dwork = np.zeros(1, dtype=np.float64)

    info = ctypes.c_int(0)

    lib.mb02xd.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.mb02xd.restype = None

    lib.mb02xd(
        b"S", b"F", b"U", None,
        ctypes.byref(ctypes.c_int(m)),
        ctypes.byref(ctypes.c_int(n)),
        ctypes.byref(ctypes.c_int(nrhs)),
        None, ctypes.byref(ctypes.c_int(0)),
        None, ctypes.byref(ctypes.c_int(0)),
        A, ctypes.byref(ctypes.c_int(lda)),
        B, ctypes.byref(ctypes.c_int(ldb)),
        ata, ctypes.byref(ctypes.c_int(ldata)),
        dwork, ctypes.byref(ctypes.c_int(1)),
        ctypes.byref(info)
    )

    assert info.value > 0, f"Expected info > 0 for singular matrix, got {info.value}"


def test_mb02xd_parameter_errors():
    """
    Test MB02XD parameter validation.
    """
    lib = get_slicot_lib()

    m, n, nrhs = 4, 3, 2
    A = np.ones((m, n), order='F', dtype=np.float64)
    B = np.ones((n, nrhs), order='F', dtype=np.float64)
    ata = np.zeros((n, n), order='F', dtype=np.float64)
    dwork = np.zeros(1, dtype=np.float64)

    lib.mb02xd.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.mb02xd.restype = None

    info = ctypes.c_int(0)
    lib.mb02xd(
        b"X", b"F", b"U", None,
        ctypes.byref(ctypes.c_int(m)),
        ctypes.byref(ctypes.c_int(n)),
        ctypes.byref(ctypes.c_int(nrhs)),
        None, ctypes.byref(ctypes.c_int(0)),
        None, ctypes.byref(ctypes.c_int(0)),
        A, ctypes.byref(ctypes.c_int(m)),
        B, ctypes.byref(ctypes.c_int(n)),
        ata, ctypes.byref(ctypes.c_int(n)),
        dwork, ctypes.byref(ctypes.c_int(1)),
        ctypes.byref(info)
    )
    assert info.value == -1, f"Expected info=-1 for invalid FORM, got {info.value}"

    info = ctypes.c_int(0)
    lib.mb02xd(
        b"S", b"X", b"U", None,
        ctypes.byref(ctypes.c_int(m)),
        ctypes.byref(ctypes.c_int(n)),
        ctypes.byref(ctypes.c_int(nrhs)),
        None, ctypes.byref(ctypes.c_int(0)),
        None, ctypes.byref(ctypes.c_int(0)),
        A, ctypes.byref(ctypes.c_int(m)),
        B, ctypes.byref(ctypes.c_int(n)),
        ata, ctypes.byref(ctypes.c_int(n)),
        dwork, ctypes.byref(ctypes.c_int(1)),
        ctypes.byref(info)
    )
    assert info.value == -2, f"Expected info=-2 for invalid STOR, got {info.value}"

    info = ctypes.c_int(0)
    lib.mb02xd(
        b"S", b"F", b"U", None,
        ctypes.byref(ctypes.c_int(-1)),
        ctypes.byref(ctypes.c_int(n)),
        ctypes.byref(ctypes.c_int(nrhs)),
        None, ctypes.byref(ctypes.c_int(0)),
        None, ctypes.byref(ctypes.c_int(0)),
        A, ctypes.byref(ctypes.c_int(m)),
        B, ctypes.byref(ctypes.c_int(n)),
        ata, ctypes.byref(ctypes.c_int(n)),
        dwork, ctypes.byref(ctypes.c_int(1)),
        ctypes.byref(info)
    )
    assert info.value == -5, f"Expected info=-5 for M<0, got {info.value}"


def test_mb02xd_residual_property():
    """
    Mathematical property test: verify residual A'*A*X - B is small.

    This validates that the solution X truly satisfies the normal equations.

    Random seed: 999 (for reproducibility)
    """
    lib = get_slicot_lib()

    m, n, nrhs = 10, 5, 3

    np.random.seed(999)
    A = np.random.randn(m, n).astype(np.float64, order='F')
    B_orig = np.random.randn(n, nrhs).astype(np.float64, order='F')
    B = B_orig.copy(order='F')

    AtA = A.T @ A

    lda = m
    ldb = n
    ldata = n
    ata = np.zeros((n, n), order='F', dtype=np.float64)
    dwork = np.zeros(1, dtype=np.float64)

    info = ctypes.c_int(0)

    lib.mb02xd.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.mb02xd.restype = None

    lib.mb02xd(
        b"S", b"F", b"U", None,
        ctypes.byref(ctypes.c_int(m)),
        ctypes.byref(ctypes.c_int(n)),
        ctypes.byref(ctypes.c_int(nrhs)),
        None, ctypes.byref(ctypes.c_int(0)),
        None, ctypes.byref(ctypes.c_int(0)),
        A, ctypes.byref(ctypes.c_int(lda)),
        B, ctypes.byref(ctypes.c_int(ldb)),
        ata, ctypes.byref(ctypes.c_int(ldata)),
        dwork, ctypes.byref(ctypes.c_int(1)),
        ctypes.byref(info)
    )

    assert info.value == 0, f"MB02XD failed with info={info.value}"

    X = B
    residual = AtA @ X - B_orig

    residual_norm = np.linalg.norm(residual, 'fro')
    B_norm = np.linalg.norm(B_orig, 'fro')

    relative_residual = residual_norm / B_norm
    assert relative_residual < 1e-13, f"Relative residual {relative_residual} too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
