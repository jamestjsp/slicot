"""
Tests for SB10UD - SVD transformation for H2 controller normalization

Tests normalization of D12 and D21 matrices to unit diagonal form.
This is a dependency-only routine tested via ctypes.
"""

import ctypes
import numpy as np
import pytest
import os


def get_slicot_lib():
    """Load the slicot shared library."""
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
    pytest.skip("slicot library not found")


def call_sb10ud(n, m, np_, ncon, nmeas, b, c, d, tu, ty, tol=0.0):
    """Call sb10ud via ctypes."""
    lib = get_slicot_lib()

    sb10ud = lib.sb10ud
    sb10ud.argtypes = [
        ctypes.c_int,   # n
        ctypes.c_int,   # m
        ctypes.c_int,   # np
        ctypes.c_int,   # ncon
        ctypes.c_int,   # nmeas
        ctypes.POINTER(ctypes.c_double),  # b
        ctypes.c_int,   # ldb
        ctypes.POINTER(ctypes.c_double),  # c
        ctypes.c_int,   # ldc
        ctypes.POINTER(ctypes.c_double),  # d
        ctypes.c_int,   # ldd
        ctypes.POINTER(ctypes.c_double),  # tu
        ctypes.c_int,   # ldtu
        ctypes.POINTER(ctypes.c_double),  # ty
        ctypes.c_int,   # ldty
        ctypes.POINTER(ctypes.c_double),  # rcond
        ctypes.c_double,  # tol
        ctypes.POINTER(ctypes.c_double),  # dwork
        ctypes.c_int,   # ldwork
        ctypes.POINTER(ctypes.c_int),  # info
    ]
    sb10ud.restype = None

    ldb = max(1, n)
    ldc = max(1, np_)
    ldd = max(1, np_)
    ldtu = max(1, ncon)
    ldty = max(1, nmeas)

    m1 = m - ncon
    m2 = ncon
    np1 = np_ - nmeas
    np2 = nmeas
    q = max(m1, m2, np1, np2, 1)
    max_n_5 = max(n, 5)
    ldwork = max(1, q * (q + max_n_5 + 1))

    dwork = np.zeros(ldwork, dtype=float, order='F')
    rcond = np.zeros(2, dtype=float, order='F')
    info = ctypes.c_int(0)

    sb10ud(
        n, m, np_, ncon, nmeas,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldb,
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldc,
        d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldd,
        tu.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldtu,
        ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldty,
        rcond.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        tol,
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )

    return info.value, rcond


class TestSB10UDBasic:
    """Basic functionality tests."""

    def test_basic_2x2_system(self):
        """
        Test basic 2x2 system with full rank D12 and D21.

        System: N=2, M=2, NP=2
        Partitioning: NCON=1 (M2=1), NMEAS=1 (NP2=1)
        """
        n, m, np_ = 2, 2, 2
        ncon, nmeas = 1, 1

        b = np.array([
            [1.0, 0.3],
            [0.5, 0.8]
        ], dtype=float, order='F')

        c = np.array([
            [0.6, 0.4],
            [0.2, 0.9]
        ], dtype=float, order='F')

        d = np.array([
            [0.0, 1.5],
            [2.0, 0.7]
        ], dtype=float, order='F')

        tu = np.zeros((ncon, ncon), dtype=float, order='F')
        ty = np.zeros((nmeas, nmeas), dtype=float, order='F')

        info, rcond = call_sb10ud(n, m, np_, ncon, nmeas, b, c, d, tu, ty)

        assert info == 0, f"Expected info=0, got {info}"
        assert rcond[0] > 0.0, "RCOND(1) should be positive"
        assert rcond[1] > 0.0, "RCOND(2) should be positive"
        assert np.abs(tu[0, 0]) > 1e-10, "TU should be non-zero"
        assert np.abs(ty[0, 0]) > 1e-10, "TY should be non-zero"

    def test_larger_system_3x4x3(self):
        """
        Test larger 3x4x3 system.

        System: N=3, M=4, NP=3
        Partitioning: NCON=2 (M2=2), NMEAS=1 (NP2=1)
        """
        n, m, np_ = 3, 4, 3
        ncon, nmeas = 2, 1

        b = np.array([
            [1.0, 0.3, 0.4, 0.2],
            [0.2, 0.8, 0.1, 0.3],
            [0.1, 0.2, 0.9, 0.5]
        ], dtype=float, order='F')

        c = np.array([
            [0.5, 0.2, 0.4],
            [0.1, 0.7, 0.3],
            [0.3, 0.2, 0.6]
        ], dtype=float, order='F')

        d = np.array([
            [0.0, 0.0, 2.0, 0.1],
            [0.0, 0.0, 0.3, 1.8],
            [1.5, 0.8, 0.4, 0.6]
        ], dtype=float, order='F')

        tu = np.zeros((ncon, ncon), dtype=float, order='F')
        ty = np.zeros((nmeas, nmeas), dtype=float, order='F')

        info, rcond = call_sb10ud(n, m, np_, ncon, nmeas, b, c, d, tu, ty)

        assert info == 0, f"Expected info=0, got {info}"
        assert rcond[0] > 0.0, "RCOND(1) should be positive"
        assert rcond[1] > 0.0, "RCOND(2) should be positive"


class TestSB10UDQuickReturn:
    """Quick return tests for edge cases."""

    def test_zero_n(self):
        """Quick return when n=0."""
        n, m, np_ = 0, 2, 2
        ncon, nmeas = 1, 1

        b = np.zeros((1, m), dtype=float, order='F')
        c = np.zeros((np_, 1), dtype=float, order='F')
        d = np.zeros((np_, m), dtype=float, order='F')
        tu = np.zeros((ncon, ncon), dtype=float, order='F')
        ty = np.zeros((nmeas, nmeas), dtype=float, order='F')

        info, rcond = call_sb10ud(n, m, np_, ncon, nmeas, b, c, d, tu, ty)

        assert info == 0
        assert rcond[0] == 1.0
        assert rcond[1] == 1.0

    def test_zero_ncon(self):
        """Quick return when ncon=0."""
        n, m, np_ = 2, 2, 2
        ncon, nmeas = 0, 1

        b = np.zeros((n, m), dtype=float, order='F')
        c = np.zeros((np_, n), dtype=float, order='F')
        d = np.zeros((np_, m), dtype=float, order='F')
        tu = np.zeros((1, 1), dtype=float, order='F')
        ty = np.zeros((nmeas, nmeas), dtype=float, order='F')

        info, rcond = call_sb10ud(n, m, np_, ncon, nmeas, b, c, d, tu, ty)

        assert info == 0
        assert rcond[0] == 1.0
        assert rcond[1] == 1.0

    def test_zero_nmeas(self):
        """Quick return when nmeas=0."""
        n, m, np_ = 2, 2, 2
        ncon, nmeas = 1, 0

        b = np.zeros((n, m), dtype=float, order='F')
        c = np.zeros((np_, n), dtype=float, order='F')
        d = np.zeros((np_, m), dtype=float, order='F')
        tu = np.zeros((ncon, ncon), dtype=float, order='F')
        ty = np.zeros((1, 1), dtype=float, order='F')

        info, rcond = call_sb10ud(n, m, np_, ncon, nmeas, b, c, d, tu, ty)

        assert info == 0
        assert rcond[0] == 1.0
        assert rcond[1] == 1.0


class TestSB10UDRankDeficient:
    """Tests for rank-deficient matrices."""

    def test_rank_deficient_d12(self):
        """D12 is rank-deficient => INFO = 1.

        With n=2, m=3, np=3, ncon=1, nmeas=1:
          M1=2, M2=1, NP1=2, NP2=1
          D12 is NP1-by-M2 = 2x1 submatrix at columns M1:M = columns 2:2
        """
        n, m, np_ = 2, 3, 3
        ncon, nmeas = 1, 1

        b = np.array([
            [1.0, 0.3, 0.2],
            [0.5, 0.8, 0.4]
        ], dtype=float, order='F')
        c = np.array([
            [0.6, 0.4],
            [0.2, 0.9],
            [0.3, 0.5]
        ], dtype=float, order='F')
        d = np.array([
            [0.0, 0.0, 0.0],  # D12 at column 2 is 0
            [0.0, 0.0, 0.0],
            [2.0, 0.8, 0.7]   # D21 at (2, 0:1), D22 at (2, 2)
        ], dtype=float, order='F')

        tu = np.zeros((ncon, ncon), dtype=float, order='F')
        ty = np.zeros((nmeas, nmeas), dtype=float, order='F')

        info, rcond = call_sb10ud(n, m, np_, ncon, nmeas, b, c, d, tu, ty)

        assert info == 1, f"Expected info=1 for rank-deficient D12, got {info}"
        assert rcond[1] == 0.0, "RCOND(2) should be 0 when INFO=1"

    def test_rank_deficient_d21(self):
        """D21 is rank-deficient => INFO = 2.

        With n=2, m=3, np=3, ncon=1, nmeas=1:
          M1=2, M2=1, NP1=2, NP2=1
          D21 is NP2-by-M1 = 1x2 submatrix at row NP1=2, columns 0:1
        """
        n, m, np_ = 2, 3, 3
        ncon, nmeas = 1, 1

        b = np.array([
            [1.0, 0.3, 0.2],
            [0.5, 0.8, 0.4]
        ], dtype=float, order='F')
        c = np.array([
            [0.6, 0.4],
            [0.2, 0.9],
            [0.3, 0.5]
        ], dtype=float, order='F')
        d = np.array([
            [0.0, 0.0, 2.0],  # D12 col at (0:1, 2)
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.7]   # D21 at (2, 0:1) is [0, 0] (rank deficient), D22 at (2,2)
        ], dtype=float, order='F')

        tu = np.zeros((ncon, ncon), dtype=float, order='F')
        ty = np.zeros((nmeas, nmeas), dtype=float, order='F')

        info, rcond = call_sb10ud(n, m, np_, ncon, nmeas, b, c, d, tu, ty)

        assert info == 2, f"Expected info=2 for rank-deficient D21, got {info}"


class TestSB10UDInvalidParameters:
    """Tests for invalid parameter detection."""

    def test_negative_n(self):
        """Negative n returns info=-1."""
        lib = get_slicot_lib()
        sb10ud = lib.sb10ud

        info = ctypes.c_int(0)
        dwork = np.zeros(10, dtype=float)
        rcond = np.zeros(2, dtype=float)
        b = np.zeros((1, 2), dtype=float, order='F')
        c = np.zeros((2, 1), dtype=float, order='F')
        d = np.zeros((2, 2), dtype=float, order='F')
        tu = np.zeros((1, 1), dtype=float, order='F')
        ty = np.zeros((1, 1), dtype=float, order='F')

        sb10ud(
            -1, 2, 2, 1, 1,
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
            d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
            tu.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
            ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
            rcond.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(0.0),
            dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 10,
            ctypes.byref(info)
        )

        assert info.value == -1

    def test_ncon_too_large(self):
        """NCON > M returns info=-4."""
        lib = get_slicot_lib()
        sb10ud = lib.sb10ud

        info = ctypes.c_int(0)
        dwork = np.zeros(10, dtype=float)
        rcond = np.zeros(2, dtype=float)
        b = np.zeros((2, 2), dtype=float, order='F')
        c = np.zeros((2, 2), dtype=float, order='F')
        d = np.zeros((2, 2), dtype=float, order='F')
        tu = np.zeros((1, 1), dtype=float, order='F')
        ty = np.zeros((1, 1), dtype=float, order='F')

        sb10ud(
            2, 2, 2, 3, 1,  # ncon=3 > m=2
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
            d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2,
            tu.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
            ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
            rcond.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(0.0),
            dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 10,
            ctypes.byref(info)
        )

        assert info.value == -4


class TestSB10UDTransformationProperties:
    """Tests for mathematical properties of the transformation."""

    def test_tu_invertible(self):
        """TU transformation matrix should be invertible."""
        n, m, np_ = 2, 3, 3
        ncon, nmeas = 2, 1

        b = np.array([
            [1.0, 0.3, 0.2],
            [0.5, 0.8, 0.4]
        ], dtype=float, order='F')

        c = np.array([
            [0.6, 0.4],
            [0.2, 0.9],
            [0.3, 0.5]
        ], dtype=float, order='F')

        d = np.array([
            [0.0, 2.0, 0.3],
            [0.0, 0.1, 1.8],
            [1.5, 0.3, 0.6]
        ], dtype=float, order='F')

        tu = np.zeros((ncon, ncon), dtype=float, order='F')
        ty = np.zeros((nmeas, nmeas), dtype=float, order='F')

        info, rcond = call_sb10ud(n, m, np_, ncon, nmeas, b, c, d, tu, ty)

        assert info == 0
        det_tu = np.linalg.det(tu)
        assert np.abs(det_tu) > 1e-10, "TU should be invertible"

    def test_rcond_consistency(self):
        """RCOND values should be consistent with singular values."""
        n, m, np_ = 2, 2, 2
        ncon, nmeas = 1, 1

        b = np.array([[1.0, 0.3], [0.5, 0.8]], dtype=float, order='F')
        c = np.array([[0.6, 0.4], [0.2, 0.9]], dtype=float, order='F')
        d = np.array([[0.0, 1.5], [2.0, 0.7]], dtype=float, order='F')

        tu = np.zeros((ncon, ncon), dtype=float, order='F')
        ty = np.zeros((nmeas, nmeas), dtype=float, order='F')

        info, rcond = call_sb10ud(n, m, np_, ncon, nmeas, b, c, d, tu, ty)

        assert info == 0
        assert 0.0 < rcond[0] <= 1.0, "RCOND(1) should be in (0, 1]"
        assert 0.0 < rcond[1] <= 1.0, "RCOND(2) should be in (0, 1]"
