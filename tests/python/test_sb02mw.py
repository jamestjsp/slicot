"""
Tests for SB02MW: Select stable eigenvalues for discrete-time Riccati.

SB02MW is a selector function for discrete-time algebraic Riccati equations.
Returns True for stable eigenvalues (modulus < 1).

Mathematical definition:
- Discrete-time stability: eigenvalue is stable if |lambda| < 1
- SB02MW selects STABLE eigenvalues: |lambda| < 1
"""

import numpy as np
from numpy.testing import assert_equal


def test_sb02mw_stable_inside_unit_circle():
    """Eigenvalues inside unit circle are stable (return True)."""
    from slicot import sb02mw

    result = sb02mw(0.0, 0.0)
    assert_equal(result, True)

    result = sb02mw(0.5, 0.0)
    assert_equal(result, True)

    result = sb02mw(0.0, 0.5)
    assert_equal(result, True)

    result = sb02mw(-0.5, 0.0)
    assert_equal(result, True)

    result = sb02mw(0.3, 0.4)
    assert_equal(result, True)

    result = sb02mw(0.6, -0.6)
    assert_equal(result, True)


def test_sb02mw_unstable_outside_unit_circle():
    """Eigenvalues outside unit circle are unstable (return False)."""
    from slicot import sb02mw

    result = sb02mw(2.0, 0.0)
    assert_equal(result, False)

    result = sb02mw(0.0, 1.5)
    assert_equal(result, False)

    result = sb02mw(0.8, 0.8)
    assert_equal(result, False)

    result = sb02mw(-1.5, 0.5)
    assert_equal(result, False)


def test_sb02mw_unstable_on_unit_circle():
    """Eigenvalues on unit circle are unstable (return False).

    Modulus = 1 is on the stability boundary, not selected as stable.
    """
    from slicot import sb02mw

    result = sb02mw(1.0, 0.0)
    assert_equal(result, False)

    result = sb02mw(0.0, 1.0)
    assert_equal(result, False)

    result = sb02mw(-1.0, 0.0)
    assert_equal(result, False)

    result = sb02mw(0.0, -1.0)
    assert_equal(result, False)

    result = sb02mw(np.sqrt(0.5), np.sqrt(0.5))
    assert_equal(result, False)

    result = sb02mw(-np.sqrt(0.5), -np.sqrt(0.5))
    assert_equal(result, False)


def test_sb02mw_boundary_near_unit_circle():
    """Test values near unit circle boundary.

    Ensures modulus < 1 comparison works correctly.
    """
    from slicot import sb02mw

    eps = np.finfo(float).eps

    result = sb02mw(1.0 + eps, 0.0)
    assert_equal(result, False)

    result = sb02mw(1.0 - eps, 0.0)
    assert_equal(result, True)


def test_sb02mw_imaginary_symmetry():
    """
    Mathematical property: sign of imaginary part doesn't affect result.

    sb02mw(reig, ieig) == sb02mw(reig, -ieig) for all values.
    Modulus |reig + i*ieig| = |reig - i*ieig|.
    """
    from slicot import sb02mw

    test_cases = [
        (0.0, 1.0),
        (0.5, 0.5),
        (-0.5, 0.8),
        (1.0, 0.5),
        (0.3, 0.4),
    ]

    for reig, ieig in test_cases:
        result_pos = sb02mw(reig, ieig)
        result_neg = sb02mw(reig, -ieig)
        assert_equal(result_pos, result_neg,
                    f"Imaginary symmetry failed for reig={reig}, ieig={ieig}")


def test_sb02mw_modulus_property():
    """
    Mathematical property: result depends only on modulus sqrt(reig^2 + ieig^2).

    All eigenvalues with same modulus should give same result.
    """
    from slicot import sb02mw

    r = 0.7
    angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
    for theta in angles:
        reig = r * np.cos(theta)
        ieig = r * np.sin(theta)
        result = sb02mw(reig, ieig)
        assert_equal(result, True, f"Expected stable for r={r}, theta={theta}")

    r = 1.5
    for theta in angles:
        reig = r * np.cos(theta)
        ieig = r * np.sin(theta)
        result = sb02mw(reig, ieig)
        assert_equal(result, False, f"Expected unstable for r={r}, theta={theta}")


def test_sb02mw_complement_sb02ms():
    """
    Mathematical property: SB02MW and SB02MS are complements.

    For any eigenvalue, exactly one of SB02MW and SB02MS returns True.
    - SB02MW returns True for stable (|lambda| < 1)
    - SB02MS returns True for unstable (|lambda| >= 1)
    """
    from slicot import sb02mw, sb02ms

    test_cases = [
        (0.0, 0.0),     # Origin: stable
        (1.0, 0.0),     # On unit circle: unstable
        (0.0, 1.0),     # On unit circle: unstable
        (0.5, 0.0),     # Inside: stable
        (2.0, 0.0),     # Outside: unstable
        (0.6, 0.6),     # Inside: stable (|0.6+0.6i| = 0.848...)
        (0.8, 0.8),     # Outside: unstable (|0.8+0.8i| = 1.131...)
    ]

    for reig, ieig in test_cases:
        result_mw = sb02mw(reig, ieig)
        result_ms = sb02ms(reig, ieig)
        assert_equal(result_mw != result_ms, True,
                    f"Complement property failed for reig={reig}, ieig={ieig}")
