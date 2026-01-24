"""
Tests for TB03AY - Polynomial matrix V(s) computation.

TB03AY is an internal helper for TB03AD that calculates a polynomial matrix V(s)
one block at a time in reverse order. It computes:
    W(s) = V2(s) * A2
    Wbar(s) = s * V:L(s) - W(s)
    V:L-1(s) = Wbar(s) * inv(R)

where R is the upper triangular part of superdiagonal blocks of A.

NOTE: TB03AY does not check inputs for errors (speed optimization).
"""

import numpy as np
import pytest
from slicot import tb03ay


def test_basic_single_block():
    """
    Test with minimal configuration: 1 block of size 1.

    Input:
        NR = 1 (total state dimension)
        INDBLK = 1 (number of blocks)
        NBLK = [1] (block sizes)
        A = 2x2 identity-like (need superdiagonal structure)

    Verifies basic operation without errors.
    """
    nr = 1
    indblk = 1
    nblk = np.array([1], dtype=np.int32)

    a = np.array([
        [2.0, 1.0],
        [0.0, 3.0]
    ], dtype=float, order='F')
    lda = 2

    ldvco1 = 1
    ldvco2 = nr
    ldpco1 = 1
    ldpco2 = 1

    vcoeff = np.zeros((ldvco1, ldvco2, indblk + 1), dtype=float, order='F')
    vcoeff[0, 0, indblk] = 1.0

    pcoeff = np.zeros((ldpco1, ldpco2, indblk + 1), dtype=float, order='F')

    vcoeff_out, pcoeff_out, info = tb03ay(
        nr, a, nblk, vcoeff, pcoeff
    )

    assert info == 0


def test_two_blocks():
    """
    Test with 2 blocks.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)

    nr = 3
    indblk = 2
    nblk = np.array([2, 1], dtype=np.int32)

    a = np.zeros((nr, nr), dtype=float, order='F')
    a[0, 0] = 1.0
    a[1, 1] = 2.0
    a[0, 2] = 0.5
    a[1, 2] = 0.5
    a[2, 2] = 3.0

    ldvco1 = 2
    ldvco2 = nr
    ldpco1 = 2
    ldpco2 = 2

    vcoeff = np.zeros((ldvco1, ldvco2, indblk + 1), dtype=float, order='F')
    vcoeff[0, 2, indblk] = 1.0

    pcoeff = np.zeros((ldpco1, ldpco2, indblk + 1), dtype=float, order='F')

    vcoeff_out, pcoeff_out, info = tb03ay(
        nr, a, nblk, vcoeff, pcoeff
    )

    assert info == 0


def test_three_blocks_upper_hessenberg():
    """
    Test with 3 blocks and upper block Hessenberg structure.

    Creates a matrix A with the structure expected from TB03AD:
    - Upper block Hessenberg form
    - Superdiagonal blocks are upper triangular

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)

    nblk = np.array([2, 2, 1], dtype=np.int32)
    nr = np.sum(nblk)
    indblk = len(nblk)

    a = np.zeros((nr, nr), dtype=float, order='F')
    for i in range(nr):
        a[i, i] = float(i + 1)

    a[0, 2] = 1.0
    a[1, 2] = 0.0
    a[1, 3] = 1.0

    a[2, 4] = 1.0
    a[3, 4] = 0.0

    ldvco1 = max(nblk)
    ldvco2 = nr
    ldpco1 = max(nblk)
    ldpco2 = max(nblk)

    vcoeff = np.zeros((ldvco1, ldvco2, indblk + 1), dtype=float, order='F')
    vcoeff[0, nr - 1, indblk] = 1.0

    pcoeff = np.zeros((ldpco1, ldpco2, indblk + 1), dtype=float, order='F')

    vcoeff_out, pcoeff_out, info = tb03ay(
        nr, a, nblk, vcoeff, pcoeff
    )

    assert info == 0


def test_singular_r_block_error():
    """
    Test error detection when R block has zero diagonal.

    When a diagonal element of the upper triangular R block is zero,
    INFO should return the index (1-based) of the zero diagonal element.
    """
    nr = 2
    indblk = 2
    nblk = np.array([1, 1], dtype=np.int32)

    a = np.zeros((nr, nr), dtype=float, order='F')
    a[0, 0] = 1.0
    a[1, 1] = 2.0

    ldvco1 = 1
    ldvco2 = nr
    ldpco1 = 1
    ldpco2 = 1

    vcoeff = np.zeros((ldvco1, ldvco2, indblk + 1), dtype=float, order='F')
    vcoeff[0, 1, indblk] = 1.0

    pcoeff = np.zeros((ldpco1, ldpco2, indblk + 1), dtype=float, order='F')

    vcoeff_out, pcoeff_out, info = tb03ay(
        nr, a, nblk, vcoeff, pcoeff
    )

    assert info > 0


def test_identity_transformation():
    """
    Test with identity-like structure where V should pass through cleanly.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)

    nblk = np.array([1, 1], dtype=np.int32)
    nr = 2
    indblk = 2

    a = np.array([
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=float, order='F')

    ldvco1 = 1
    ldvco2 = nr
    ldpco1 = 1
    ldpco2 = 1

    vcoeff = np.zeros((ldvco1, ldvco2, indblk + 1), dtype=float, order='F')
    vcoeff[0, 1, indblk] = 1.0

    pcoeff = np.zeros((ldpco1, ldpco2, indblk + 1), dtype=float, order='F')

    vcoeff_out, pcoeff_out, info = tb03ay(
        nr, a, nblk, vcoeff, pcoeff
    )

    assert info == 0
    assert vcoeff_out[0, 0, 1] != 0.0 or vcoeff_out[0, 0, 2] != 0.0


def test_nr_zero():
    """
    Test edge case with NR = 0 (empty state dimension).
    """
    nr = 0
    indblk = 0
    nblk = np.array([], dtype=np.int32)

    a = np.zeros((1, 1), dtype=float, order='F')

    ldvco1 = 1
    ldvco2 = 1
    ldpco1 = 1
    ldpco2 = 1

    vcoeff = np.zeros((ldvco1, ldvco2, 1), dtype=float, order='F')
    pcoeff = np.zeros((ldpco1, ldpco2, 1), dtype=float, order='F')

    vcoeff_out, pcoeff_out, info = tb03ay(
        nr, a, nblk, vcoeff, pcoeff
    )

    assert info == 0


def test_polynomial_structure():
    """
    Test that output maintains proper polynomial coefficient structure.

    The algorithm computes:
        Wbar(s) = s * V:L(s) - W(s)
    which shifts polynomial coefficients by one degree.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)

    nblk = np.array([1, 1, 1], dtype=np.int32)
    nr = 3
    indblk = 3

    a = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 2.0, 1.0],
        [0.0, 0.0, 3.0]
    ], dtype=float, order='F')

    ldvco1 = 1
    ldvco2 = nr
    ldpco1 = 1
    ldpco2 = 1

    vcoeff = np.zeros((ldvco1, ldvco2, indblk + 1), dtype=float, order='F')
    vcoeff[0, nr - 1, indblk] = 1.0

    pcoeff = np.zeros((ldpco1, ldpco2, indblk + 1), dtype=float, order='F')

    vcoeff_out, pcoeff_out, info = tb03ay(
        nr, a, nblk, vcoeff, pcoeff
    )

    assert info == 0

    has_nonzero = False
    for k in range(indblk + 1):
        for j in range(nr):
            if abs(vcoeff_out[0, j, k]) > 1e-15:
                has_nonzero = True
                break
    assert has_nonzero


def test_larger_blocks():
    """
    Test with larger block sizes.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)

    nblk = np.array([2, 2], dtype=np.int32)
    nr = 4
    indblk = 2

    a = np.zeros((nr, nr), dtype=float, order='F')
    for i in range(nr):
        a[i, i] = float(i + 1)
    a[0, 2] = 1.0
    a[1, 3] = 1.0

    ldvco1 = 2
    ldvco2 = nr
    ldpco1 = 2
    ldpco2 = 2

    vcoeff = np.zeros((ldvco1, ldvco2, indblk + 1), dtype=float, order='F')
    vcoeff[0, 2, indblk] = 1.0
    vcoeff[1, 3, indblk] = 1.0

    pcoeff = np.zeros((ldpco1, ldpco2, indblk + 1), dtype=float, order='F')

    vcoeff_out, pcoeff_out, info = tb03ay(
        nr, a, nblk, vcoeff, pcoeff
    )

    assert info == 0
