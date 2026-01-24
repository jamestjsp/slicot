"""
Tests for SB16AY - Cholesky factors of frequency-weighted controllability
and observability Grammians for controller reduction.

SB16AY computes for given state-space representations (A,B,C,D) and
(Ac,Bc,Cc,Dc) of the open-loop system G and feedback controller K,
the Cholesky factors S and R of the frequency-weighted Grammians
P = S*S' (controllability) and Q = R'*R (observability).

The controller must stabilize the closed-loop system.
Ac must be in block-diagonal real Schur form: Ac = diag(Ac1, Ac2),
where Ac1 contains unstable eigenvalues and Ac2 contains stable eigenvalues.
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
def sb16ay(lib):
    """Get the sb16ay function with proper signature."""
    func = lib.sb16ay
    func.argtypes = [
        ctypes.c_char_p,                  # dico
        ctypes.c_char_p,                  # jobc
        ctypes.c_char_p,                  # jobo
        ctypes.c_char_p,                  # weight
        ctypes.c_int,                     # n
        ctypes.c_int,                     # m
        ctypes.c_int,                     # p
        ctypes.c_int,                     # nc
        ctypes.c_int,                     # ncs
        ctypes.POINTER(ctypes.c_double),  # a
        ctypes.c_int,                     # lda
        ctypes.POINTER(ctypes.c_double),  # b
        ctypes.c_int,                     # ldb
        ctypes.POINTER(ctypes.c_double),  # c
        ctypes.c_int,                     # ldc
        ctypes.POINTER(ctypes.c_double),  # d
        ctypes.c_int,                     # ldd
        ctypes.POINTER(ctypes.c_double),  # ac
        ctypes.c_int,                     # ldac
        ctypes.POINTER(ctypes.c_double),  # bc
        ctypes.c_int,                     # ldbc
        ctypes.POINTER(ctypes.c_double),  # cc
        ctypes.c_int,                     # ldcc
        ctypes.POINTER(ctypes.c_double),  # dc
        ctypes.c_int,                     # lddc
        ctypes.POINTER(ctypes.c_double),  # scalec
        ctypes.POINTER(ctypes.c_double),  # scaleo
        ctypes.POINTER(ctypes.c_double),  # s
        ctypes.c_int,                     # lds
        ctypes.POINTER(ctypes.c_double),  # r
        ctypes.c_int,                     # ldr
        ctypes.POINTER(ctypes.c_int),     # iwork
        ctypes.POINTER(ctypes.c_double),  # dwork
        ctypes.c_int,                     # ldwork
        ctypes.POINTER(ctypes.c_int),     # info
    ]
    func.restype = None
    return func


def test_no_weighting_continuous(sb16ay):
    """
    Test WEIGHT='N' for continuous-time.

    When no weighting is used:
    - Controllability Grammian P solves: Ac2*P + P*Ac2' + scalec^2*Bc2*Bc2' = 0
    - Observability Grammian Q solves: Ac2'*Q + Q*Ac2 + scaleo^2*Cc2'*Cc2 = 0

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    n, m, p = 2, 1, 1
    nc, ncs = 3, 3

    a = np.array([[-1.0, 0.5], [0.0, -2.0]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    c = np.array([[1.0, 1.0]], order='F', dtype=np.float64)
    d = np.array([[0.0]], order='F', dtype=np.float64)

    ac = np.array([
        [-0.5, 0.1, 0.0],
        [0.0, -1.0, 0.2],
        [0.0, 0.0, -1.5]
    ], order='F', dtype=np.float64)
    bc = np.array([[1.0], [0.5], [0.0]], order='F', dtype=np.float64)
    cc = np.array([[0.5, 0.3, 0.2]], order='F', dtype=np.float64)
    dc = np.array([[0.1]], order='F', dtype=np.float64)

    s = np.zeros((ncs, ncs), order='F', dtype=np.float64)
    r = np.zeros((ncs, ncs), order='F', dtype=np.float64)
    scalec = ctypes.c_double(0.0)
    scaleo = ctypes.c_double(0.0)

    ldwork = ncs * (max(m, p) + 5)
    ldwork = max(1, ldwork)
    dwork = np.zeros(ldwork, dtype=np.float64)
    iwork = np.zeros(1, dtype=np.int32)
    info = ctypes.c_int(0)

    sb16ay(
        b"C", b"S", b"S", b"N",
        n, m, p, nc, ncs,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), p,
        d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), p,
        ac.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nc,
        bc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nc,
        cc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        dc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        ctypes.byref(scalec), ctypes.byref(scaleo),
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ncs,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ncs,
        iwork.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )

    assert info.value == 0
    assert scalec.value > 0
    assert scaleo.value > 0

    s_upper = np.triu(s)
    r_upper = np.triu(r)
    p = s_upper @ s_upper.T
    q = r_upper.T @ r_upper

    bc2 = bc
    cc2 = cc

    res_c = ac @ p + p @ ac.T + scalec.value**2 * bc2 @ bc2.T
    res_o = ac.T @ q + q @ ac + scaleo.value**2 * cc2.T @ cc2

    np.testing.assert_allclose(res_c, 0.0, atol=1e-10)
    np.testing.assert_allclose(res_o, 0.0, atol=1e-10)


def test_no_weighting_discrete(sb16ay):
    """
    Test WEIGHT='N' for discrete-time.

    When no weighting is used:
    - Controllability Grammian P solves: Ac2*P*Ac2' - P + scalec^2*Bc2*Bc2' = 0
    - Observability Grammian Q solves: Ac2'*Q*Ac2 - Q + scaleo^2*Cc2'*Cc2 = 0

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    n, m, p = 2, 1, 1
    nc, ncs = 3, 3

    a = np.array([[0.5, 0.1], [0.0, 0.4]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    c = np.array([[1.0, 1.0]], order='F', dtype=np.float64)
    d = np.array([[0.0]], order='F', dtype=np.float64)

    ac = np.array([
        [0.3, 0.1, 0.0],
        [0.0, 0.4, 0.1],
        [0.0, 0.0, 0.2]
    ], order='F', dtype=np.float64)
    bc = np.array([[1.0], [0.5], [0.0]], order='F', dtype=np.float64)
    cc = np.array([[0.5, 0.3, 0.2]], order='F', dtype=np.float64)
    dc = np.array([[0.1]], order='F', dtype=np.float64)

    s = np.zeros((ncs, ncs), order='F', dtype=np.float64)
    r = np.zeros((ncs, ncs), order='F', dtype=np.float64)
    scalec = ctypes.c_double(0.0)
    scaleo = ctypes.c_double(0.0)

    ldwork = ncs * (max(m, p) + 5)
    ldwork = max(1, ldwork)
    dwork = np.zeros(ldwork, dtype=np.float64)
    iwork = np.zeros(1, dtype=np.int32)
    info = ctypes.c_int(0)

    sb16ay(
        b"D", b"S", b"S", b"N",
        n, m, p, nc, ncs,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), p,
        d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), p,
        ac.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nc,
        bc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nc,
        cc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        dc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        ctypes.byref(scalec), ctypes.byref(scaleo),
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ncs,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ncs,
        iwork.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )

    assert info.value == 0
    assert scalec.value > 0
    assert scaleo.value > 0

    s_upper = np.triu(s)
    r_upper = np.triu(r)
    p = s_upper @ s_upper.T
    q = r_upper.T @ r_upper

    bc2 = bc
    cc2 = cc

    res_c = ac @ p @ ac.T - p + scalec.value**2 * bc2 @ bc2.T
    res_o = ac.T @ q @ ac - q + scaleo.value**2 * cc2.T @ cc2

    np.testing.assert_allclose(res_c, 0.0, atol=1e-10)
    np.testing.assert_allclose(res_o, 0.0, atol=1e-10)


def test_zero_ncs(sb16ay):
    """
    Test with NCS=0 (quick return).
    """
    n, m, p = 2, 1, 1
    nc, ncs = 2, 0

    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    c = np.array([[1.0, 1.0]], order='F', dtype=np.float64)
    d = np.array([[0.0]], order='F', dtype=np.float64)

    ac = np.array([[1.5, 0.1], [0.0, 2.0]], order='F', dtype=np.float64)
    bc = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    cc = np.array([[1.0, 0.5]], order='F', dtype=np.float64)
    dc = np.array([[0.1]], order='F', dtype=np.float64)

    s = np.zeros((1, 1), order='F', dtype=np.float64)
    r = np.zeros((1, 1), order='F', dtype=np.float64)
    scalec = ctypes.c_double(0.0)
    scaleo = ctypes.c_double(0.0)

    ldwork = 1
    dwork = np.zeros(ldwork, dtype=np.float64)
    iwork = np.zeros(1, dtype=np.int32)
    info = ctypes.c_int(0)

    sb16ay(
        b"C", b"S", b"S", b"N",
        n, m, p, nc, ncs,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), p,
        d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), p,
        ac.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nc,
        bc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nc,
        cc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        dc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        ctypes.byref(scalec), ctypes.byref(scaleo),
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        iwork.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )

    assert info.value == 0
    assert scalec.value == pytest.approx(1.0)
    assert scaleo.value == pytest.approx(1.0)


def test_invalid_dico(sb16ay):
    """Test invalid DICO parameter returns info=-1."""
    n, m, p = 2, 1, 1
    nc, ncs = 2, 2

    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    c = np.array([[1.0, 1.0]], order='F', dtype=np.float64)
    d = np.array([[0.0]], order='F', dtype=np.float64)
    ac = np.array([[-0.5, 0.0], [0.0, -1.0]], order='F', dtype=np.float64)
    bc = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    cc = np.array([[1.0, 0.5]], order='F', dtype=np.float64)
    dc = np.array([[0.1]], order='F', dtype=np.float64)

    s = np.zeros((ncs, ncs), order='F', dtype=np.float64)
    r = np.zeros((ncs, ncs), order='F', dtype=np.float64)
    scalec = ctypes.c_double(0.0)
    scaleo = ctypes.c_double(0.0)

    ldwork = ncs * (max(m, p) + 5)
    dwork = np.zeros(ldwork, dtype=np.float64)
    iwork = np.zeros(1, dtype=np.int32)
    info = ctypes.c_int(0)

    sb16ay(
        b"X", b"S", b"S", b"N",
        n, m, p, nc, ncs,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), p,
        d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), p,
        ac.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nc,
        bc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nc,
        cc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        dc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        ctypes.byref(scalec), ctypes.byref(scaleo),
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ncs,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ncs,
        iwork.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )

    assert info.value == -1


def test_invalid_ncs(sb16ay):
    """Test NCS > NC returns info=-9."""
    n, m, p = 2, 1, 1
    nc, ncs = 2, 3

    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    c = np.array([[1.0, 1.0]], order='F', dtype=np.float64)
    d = np.array([[0.0]], order='F', dtype=np.float64)
    ac = np.array([[-0.5, 0.0], [0.0, -1.0]], order='F', dtype=np.float64)
    bc = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    cc = np.array([[1.0, 0.5]], order='F', dtype=np.float64)
    dc = np.array([[0.1]], order='F', dtype=np.float64)

    s = np.zeros((ncs, ncs), order='F', dtype=np.float64)
    r = np.zeros((ncs, ncs), order='F', dtype=np.float64)
    scalec = ctypes.c_double(0.0)
    scaleo = ctypes.c_double(0.0)

    ldwork = ncs * (max(m, p) + 5)
    dwork = np.zeros(ldwork, dtype=np.float64)
    iwork = np.zeros(1, dtype=np.int32)
    info = ctypes.c_int(0)

    sb16ay(
        b"C", b"S", b"S", b"N",
        n, m, p, nc, ncs,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), p,
        d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), p,
        ac.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nc,
        bc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nc,
        cc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        dc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        ctypes.byref(scalec), ctypes.byref(scaleo),
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ncs,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ncs,
        iwork.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )

    assert info.value == -9


def test_unstable_ac2_continuous(sb16ay):
    """
    Test that unstable Ac2 (stable part) returns info=5 for continuous.
    """
    n, m, p = 2, 1, 1
    nc, ncs = 2, 2

    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    c = np.array([[1.0, 1.0]], order='F', dtype=np.float64)
    d = np.array([[0.0]], order='F', dtype=np.float64)

    ac = np.array([[1.0, 0.0], [0.0, 2.0]], order='F', dtype=np.float64)
    bc = np.array([[1.0], [0.0]], order='F', dtype=np.float64)
    cc = np.array([[1.0, 0.5]], order='F', dtype=np.float64)
    dc = np.array([[0.1]], order='F', dtype=np.float64)

    s = np.zeros((ncs, ncs), order='F', dtype=np.float64)
    r = np.zeros((ncs, ncs), order='F', dtype=np.float64)
    scalec = ctypes.c_double(0.0)
    scaleo = ctypes.c_double(0.0)

    ldwork = ncs * (max(m, p) + 5)
    dwork = np.zeros(ldwork, dtype=np.float64)
    iwork = np.zeros(1, dtype=np.int32)
    info = ctypes.c_int(0)

    sb16ay(
        b"C", b"S", b"S", b"N",
        n, m, p, nc, ncs,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), p,
        d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), p,
        ac.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nc,
        bc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nc,
        cc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        dc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        ctypes.byref(scalec), ctypes.byref(scaleo),
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ncs,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ncs,
        iwork.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )

    assert info.value == 5


def test_cholesky_symmetry_continuous(sb16ay):
    """
    Test that S*S' is symmetric positive semi-definite (controllability)
    and R'*R is symmetric positive semi-definite (observability).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    n, m, p = 2, 1, 1
    nc, ncs = 4, 4

    a = np.array([[-1.0, 0.2], [-0.1, -2.0]], order='F', dtype=np.float64)
    b = np.array([[1.0], [0.5]], order='F', dtype=np.float64)
    c = np.array([[1.0, 0.5]], order='F', dtype=np.float64)
    d = np.array([[0.0]], order='F', dtype=np.float64)

    ac = np.array([
        [-0.5, 0.1, 0.0, 0.0],
        [0.0, -0.8, 0.1, 0.0],
        [0.0, 0.0, -1.2, 0.1],
        [0.0, 0.0, 0.0, -1.5]
    ], order='F', dtype=np.float64)
    bc = np.array([[1.0], [0.5], [0.3], [0.1]], order='F', dtype=np.float64)
    cc = np.array([[0.5, 0.4, 0.3, 0.2]], order='F', dtype=np.float64)
    dc = np.array([[0.05]], order='F', dtype=np.float64)

    s = np.zeros((ncs, ncs), order='F', dtype=np.float64)
    r = np.zeros((ncs, ncs), order='F', dtype=np.float64)
    scalec = ctypes.c_double(0.0)
    scaleo = ctypes.c_double(0.0)

    ldwork = ncs * (max(m, p) + 5)
    dwork = np.zeros(ldwork, dtype=np.float64)
    iwork = np.zeros(1, dtype=np.int32)
    info = ctypes.c_int(0)

    sb16ay(
        b"C", b"S", b"S", b"N",
        n, m, p, nc, ncs,
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), p,
        d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), p,
        ac.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nc,
        bc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nc,
        cc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        dc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        ctypes.byref(scalec), ctypes.byref(scaleo),
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ncs,
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ncs,
        iwork.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dwork.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ldwork,
        ctypes.byref(info)
    )

    assert info.value == 0

    s_upper = np.triu(s)
    r_upper = np.triu(r)
    p = s_upper @ s_upper.T
    q = r_upper.T @ r_upper

    np.testing.assert_allclose(p, p.T, rtol=1e-14)
    np.testing.assert_allclose(q, q.T, rtol=1e-14)

    eig_p = np.linalg.eigvalsh(p)
    eig_q = np.linalg.eigvalsh(q)
    assert np.all(eig_p >= -1e-14)
    assert np.all(eig_q >= -1e-14)
