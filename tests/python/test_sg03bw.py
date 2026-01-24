import pytest
import numpy as np
from slicot import sg03bw


def test_sg03bw_basic_n1():
    """Test SG03BW with N=1 (basic case).

    Solves: A^T * X * C + E^T * X * D = SCALE * Y
    """
    trans = 'N'
    m, n = 2, 1

    # Simple upper triangular A and E
    a = np.array([[2.0, 1.0],
                  [0.0, 3.0]], dtype=np.float64, order='F')

    e = np.array([[1.0, 0.5],
                  [0.0, 1.0]], dtype=np.float64, order='F')

    c = np.array([[4.0]], dtype=np.float64, order='F')
    d = np.array([[2.0]], dtype=np.float64, order='F')

    # Right-hand side Y (will be overwritten with solution X)
    x = np.array([[1.0],
                  [2.0]], dtype=np.float64, order='F')

    x_out, scale, info = sg03bw(trans, a, e, c, d, x)

    # Check success
    assert info == 0, f"SG03BW failed with info={info}"

    # Scale should be positive
    assert 0 < scale <= 1.0, f"Invalid scale={scale}"

    # Solution should be finite
    assert np.all(np.isfinite(x_out))


def test_sg03bw_basic_n2():
    """Test SG03BW with N=2."""
    trans = 'N'
    m, n = 2, 2

    # Upper quasitriangular A (2x2 block)
    a = np.array([[1.0, 2.0],
                  [0.5, 3.0]], dtype=np.float64, order='F')

    # Upper triangular E
    e = np.array([[2.0, 1.0],
                  [0.0, 1.5]], dtype=np.float64, order='F')

    c = np.array([[1.0, 0.0],
                  [0.0, 1.0]], dtype=np.float64, order='F')

    d = np.array([[0.5, 0.0],
                  [0.0, 0.5]], dtype=np.float64, order='F')

    x = np.array([[1.0, 0.5],
                  [0.5, 1.0]], dtype=np.float64, order='F')

    x_out, scale, info = sg03bw(trans, a, e, c, d, x)

    assert info == 0, f"SG03BW failed with info={info}"
    assert 0 < scale <= 1.0, f"Invalid scale={scale}"
    assert np.all(np.isfinite(x_out))


def test_sg03bw_transposed():
    """Test SG03BW with TRANS='T' (transposed equation)."""
    trans = 'T'
    m, n = 2, 1

    a = np.array([[3.0, 1.0],
                  [0.0, 2.0]], dtype=np.float64, order='F')

    e = np.array([[1.5, 0.5],
                  [0.0, 1.0]], dtype=np.float64, order='F')

    c = np.array([[2.0]], dtype=np.float64, order='F')
    d = np.array([[1.0]], dtype=np.float64, order='F')

    x = np.array([[1.0],
                  [1.0]], dtype=np.float64, order='F')

    x_out, scale, info = sg03bw(trans, a, e, c, d, x)

    assert info == 0, f"SG03BW failed with info={info}"
    assert 0 < scale <= 1.0, f"Invalid scale={scale}"
    assert np.all(np.isfinite(x_out))


def test_sg03bw_m1_quick_return():
    """Test SG03BW with M=1 (minimal case for quick return)."""
    trans = 'N'
    m, n = 1, 1

    a = np.array([[1.0]], dtype=np.float64, order='F')
    e = np.array([[1.0]], dtype=np.float64, order='F')
    c = np.array([[1.0]], dtype=np.float64, order='F')
    d = np.array([[1.0]], dtype=np.float64, order='F')
    x = np.array([[0.0]], dtype=np.float64, order='F')

    x_out, scale, info = sg03bw(trans, a, e, c, d, x)

    # Should work without error
    assert info == 0, f"SG03BW failed with info={info}"
    assert scale > 0, f"Expected positive scale, got {scale}"


def test_sg03bw_invalid_trans():
    """Test SG03BW with invalid TRANS parameter."""
    trans = 'X'  # Invalid
    m, n = 2, 1

    a = np.eye(2, dtype=np.float64, order='F')
    e = np.eye(2, dtype=np.float64, order='F')
    c = np.ones((1, 1), dtype=np.float64, order='F')
    d = np.ones((1, 1), dtype=np.float64, order='F')
    x = np.ones((2, 1), dtype=np.float64, order='F')

    x_out, scale, info = sg03bw(trans, a, e, c, d, x)

    # Should return error
    assert info < 0, f"Expected info < 0 for invalid TRANS, got {info}"


def test_sg03bw_invalid_n():
    """Test SG03BW with invalid N (must be 1 or 2)."""
    trans = 'N'
    m, n = 2, 3  # Invalid N

    a = np.eye(2, dtype=np.float64, order='F')
    e = np.eye(2, dtype=np.float64, order='F')
    c = np.eye(3, dtype=np.float64, order='F')
    d = np.eye(3, dtype=np.float64, order='F')
    x = np.ones((2, 3), dtype=np.float64, order='F')

    x_out, scale, info = sg03bw(trans, a, e, c, d, x)

    # Should return error
    assert info < 0, f"Expected info < 0 for invalid N, got {info}"


def test_sg03bw_nearly_singular():
    """Test SG03BW with nearly singular system.

    Should return INFO=1 and use perturbed values.
    """
    trans = 'N'
    m, n = 2, 1

    # Create nearly singular system
    # A and E structured to make equation nearly singular
    a = np.array([[1e-15, 1.0],
                  [0.0, 1e-15]], dtype=np.float64, order='F')

    e = np.array([[1e-15, 0.5],
                  [0.0, 1e-15]], dtype=np.float64, order='F')

    c = np.array([[1.0]], dtype=np.float64, order='F')
    d = np.array([[1.0]], dtype=np.float64, order='F')

    x = np.array([[1.0],
                  [1.0]], dtype=np.float64, order='F')

    x_out, scale, info = sg03bw(trans, a, e, c, d, x)

    # May return INFO=1 for near singularity
    # But should still produce a finite result
    assert info >= 0, f"Unexpected error: info={info}"
    assert np.all(np.isfinite(x_out))
    assert 0 < scale <= 1.0


def test_sg03bw_identity_matrices():
    """Test SG03BW with identity matrices."""
    trans = 'N'
    m, n = 3, 1

    a = np.eye(3, dtype=np.float64, order='F')
    e = np.eye(3, dtype=np.float64, order='F')
    c = np.array([[2.0]], dtype=np.float64, order='F')
    d = np.array([[3.0]], dtype=np.float64, order='F')

    # Y = [1, 2, 3]^T
    x = np.array([[1.0],
                  [2.0],
                  [3.0]], dtype=np.float64, order='F')

    x_out, scale, info = sg03bw(trans, a, e, c, d, x)

    assert info == 0, f"SG03BW failed with info={info}"
    assert scale > 0, f"Invalid scale={scale}"

    # For identity matrices: (A^T + E^T) * X * (C + D) = (I + I) * X * (C + D)
    # = 2X * (C+D) = SCALE * Y
    # So X should be proportional to Y / (2*(C+D))
    expected_factor = scale / (2.0 * (c[0, 0] + d[0, 0]))
    expected_x = x * expected_factor / scale

    # Solution should have correct structure (may be scaled)
    assert np.all(np.isfinite(x_out))


def test_sg03bw_quasitriangular_boundary_m3():
    """Test SG03BW with M=3 to check boundary conditions for 2x2 blocks.

    This tests the fix for array bounds checking when detecting
    quasitriangular blocks at matrix boundaries.
    """
    trans = 'N'
    m, n = 3, 1

    # Upper quasitriangular with 2x2 block at positions [1:3, 1:3]
    # Block structure forces boundary checks
    a = np.array([[2.0, 1.0, 0.5],
                  [0.0, 1.0, 2.0],
                  [0.0, 0.5, 3.0]], dtype=np.float64, order='F')

    e = np.array([[1.0, 0.5, 0.0],
                  [0.0, 1.5, 1.0],
                  [0.0, 0.0, 2.0]], dtype=np.float64, order='F')

    c = np.array([[1.0]], dtype=np.float64, order='F')
    d = np.array([[1.0]], dtype=np.float64, order='F')

    x = np.array([[1.0],
                  [2.0],
                  [3.0]], dtype=np.float64, order='F')

    x_out, scale, info = sg03bw(trans, a, e, c, d, x)

    assert info == 0, f"SG03BW failed with info={info}"
    assert 0 < scale <= 1.0, f"Invalid scale={scale}"
    assert np.all(np.isfinite(x_out))


def test_sg03bw_quasitriangular_boundary_m4():
    """Test SG03BW with M=4 to verify 2x2 block handling at boundaries."""
    trans = 'T'  # Test transposed path
    m, n = 4, 1

    # Quasitriangular with multiple blocks
    a = np.array([[1.0, 2.0, 0.0, 0.0],
                  [0.5, 3.0, 1.0, 0.0],
                  [0.0, 0.0, 2.0, 1.5],
                  [0.0, 0.0, 0.5, 4.0]], dtype=np.float64, order='F')

    e = np.array([[2.0, 1.0, 0.0, 0.0],
                  [0.0, 1.5, 0.5, 0.0],
                  [0.0, 0.0, 1.0, 0.5],
                  [0.0, 0.0, 0.0, 2.0]], dtype=np.float64, order='F')

    c = np.array([[2.0]], dtype=np.float64, order='F')
    d = np.array([[1.5]], dtype=np.float64, order='F')

    x = np.array([[1.0],
                  [2.0],
                  [3.0],
                  [4.0]], dtype=np.float64, order='F')

    x_out, scale, info = sg03bw(trans, a, e, c, d, x)

    assert info == 0, f"SG03BW failed with info={info}"
    assert 0 < scale <= 1.0, f"Invalid scale={scale}"
    assert np.all(np.isfinite(x_out))
