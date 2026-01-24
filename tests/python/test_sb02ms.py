"""
Tests for SB02MS: Select unstable eigenvalues for discrete-time Riccati.

SB02MS is a selector function for discrete-time algebraic Riccati equations.
Returns True for unstable eigenvalues (modulus >= 1).

Mathematical definition:
- Discrete-time stability: eigenvalue is stable if |lambda| < 1
- SB02MS selects UNSTABLE eigenvalues: |lambda| >= 1
"""

import numpy as np
from numpy.testing import assert_equal


def test_sb02ms_unstable_outside_unit_circle():
    """Eigenvalues outside unit circle are unstable (return True)."""
    from slicot import sb02ms

    result = sb02ms(2.0, 0.0)
    assert_equal(result, True)

    result = sb02ms(0.0, 1.5)
    assert_equal(result, True)

    result = sb02ms(0.8, 0.8)
    assert_equal(result, True)

    result = sb02ms(-1.5, 0.5)
    assert_equal(result, True)


def test_sb02ms_unstable_on_unit_circle():
    """Eigenvalues on unit circle are unstable (return True).

    Modulus = 1 is on the stability boundary, selected as unstable.
    """
    from slicot import sb02ms

    result = sb02ms(1.0, 0.0)
    assert_equal(result, True)

    result = sb02ms(0.0, 1.0)
    assert_equal(result, True)

    result = sb02ms(-1.0, 0.0)
    assert_equal(result, True)

    result = sb02ms(0.0, -1.0)
    assert_equal(result, True)

    result = sb02ms(np.sqrt(0.5), np.sqrt(0.5))
    assert_equal(result, True)

    result = sb02ms(-np.sqrt(0.5), -np.sqrt(0.5))
    assert_equal(result, True)


def test_sb02ms_stable_inside_unit_circle():
    """Eigenvalues inside unit circle are stable (return False)."""
    from slicot import sb02ms

    result = sb02ms(0.0, 0.0)
    assert_equal(result, False)

    result = sb02ms(0.5, 0.0)
    assert_equal(result, False)

    result = sb02ms(0.0, 0.5)
    assert_equal(result, False)

    result = sb02ms(-0.5, 0.0)
    assert_equal(result, False)

    result = sb02ms(0.3, 0.4)
    assert_equal(result, False)

    result = sb02ms(0.6, -0.6)
    assert_equal(result, False)


def test_sb02ms_boundary_near_unit_circle():
    """Test values near unit circle boundary.

    Ensures modulus >= 1 comparison works correctly.
    """
    from slicot import sb02ms

    eps = np.finfo(float).eps

    result = sb02ms(1.0 + eps, 0.0)
    assert_equal(result, True)

    result = sb02ms(1.0 - eps, 0.0)
    assert_equal(result, False)


def test_sb02ms_imaginary_symmetry():
    """
    Mathematical property: sign of imaginary part doesn't affect result.

    sb02ms(reig, ieig) == sb02ms(reig, -ieig) for all values.
    Modulus |reig + i*ieig| = |reig - i*ieig|.
    """
    from slicot import sb02ms

    test_cases = [
        (0.0, 1.0),
        (0.5, 0.5),
        (-0.5, 0.8),
        (1.0, 0.5),
        (0.3, 0.4),
    ]

    for reig, ieig in test_cases:
        result_pos = sb02ms(reig, ieig)
        result_neg = sb02ms(reig, -ieig)
        assert_equal(result_pos, result_neg,
                    f"Imaginary symmetry failed for reig={reig}, ieig={ieig}")


def test_sb02ms_modulus_property():
    """
    Mathematical property: result depends only on modulus sqrt(reig^2 + ieig^2).

    All eigenvalues with same modulus should give same result.
    """
    from slicot import sb02ms

    r = 0.7
    angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
    for theta in angles:
        reig = r * np.cos(theta)
        ieig = r * np.sin(theta)
        result = sb02ms(reig, ieig)
        assert_equal(result, False, f"Expected stable for r={r}, theta={theta}")

    r = 1.5
    for theta in angles:
        reig = r * np.cos(theta)
        ieig = r * np.sin(theta)
        result = sb02ms(reig, ieig)
        assert_equal(result, True, f"Expected unstable for r={r}, theta={theta}")


def test_sb02ms_complement_sb02mv():
    """
    Mathematical property: SB02MS and SB02MV are complements.

    SB02MS selects |lambda| >= 1 (unstable discrete-time)
    SB02MV selects |lambda| < 1 (stable discrete-time)
    """
    from slicot import sb02ms

    test_cases = [
        (0.0, 0.0, False),    # Origin: stable
        (1.0, 0.0, True),     # On unit circle: unstable
        (0.0, 1.0, True),     # On unit circle: unstable
        (0.5, 0.0, False),    # Inside: stable
        (2.0, 0.0, True),     # Outside: unstable
        (0.6, 0.6, False),    # Inside: stable (|0.6+0.6i| = 0.848...)
        (0.8, 0.8, True),     # Outside: unstable (|0.8+0.8i| = 1.131...)
    ]

    for reig, ieig, expected_unstable in test_cases:
        result = sb02ms(reig, ieig)
        assert_equal(result, expected_unstable,
                    f"Failed for reig={reig}, ieig={ieig}")
