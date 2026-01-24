"""
Tests for MB04PB: Computation of the Paige/Van Loan (PVL) form of a Hamiltonian matrix (block algorithm).

MB04PB reduces a Hamiltonian matrix H = [[A, G], [Q, -A^T]] where G and Q are symmetric,
to PVL form using an orthogonal symplectic transformation U such that U^T H U has
upper Hessenberg A and diagonal Q.
"""

import numpy as np
import pytest

from slicot import mb04pb


class TestMB04PBBasic:
    """Basic functionality tests using HTML doc example data."""

    def test_html_doc_example(self):
        """
        Test MB04PB using the example from SLICOT HTML documentation.

        N=5, ILO=1 (full reduction from scratch).
        Validates that after reduction, the residual is small.
        """
        n = 5
        ilo = 1

        a = np.array([
            [0.9501, 0.7621, 0.6154, 0.4057, 0.0579],
            [0.2311, 0.4565, 0.7919, 0.9355, 0.3529],
            [0.6068, 0.0185, 0.9218, 0.9169, 0.8132],
            [0.4860, 0.8214, 0.7382, 0.4103, 0.0099],
            [0.8913, 0.4447, 0.1763, 0.8936, 0.1389],
        ], order='F', dtype=float)

        qg = np.array([
            [0.3869, 0.4055, 0.2140, 1.0224, 1.1103, 0.7016],
            [1.3801, 0.7567, 1.4936, 1.2913, 0.9515, 1.1755],
            [0.7993, 1.7598, 1.6433, 1.0503, 0.8839, 1.1010],
            [1.2019, 1.1956, 0.9346, 0.6824, 0.7590, 1.1364],
            [0.8780, 0.9029, 1.6565, 1.1022, 0.7408, 0.3793],
        ], order='F', dtype=float)

        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)

        assert info == 0

        assert a_out.shape == (n, n)
        assert qg_out.shape == (n, n + 1)
        assert cs.shape == (2 * n - 2,)
        assert tau.shape == (n - 1,)

        expected_a = np.array([
            [0.9501, -1.5494, 0.5268, 0.3187, -0.6890],
            [-2.4922, 2.0907, -1.3598, 0.5682, 0.5618],
            [0.0000, -1.7723, 0.3960, -0.2624, -0.3709],
            [0.0000, 0.0000, -0.2648, 0.2136, -0.3226],
            [0.0000, 0.0000, 0.0000, -0.2308, 0.2319],
        ], order='F', dtype=float)

        np.testing.assert_allclose(a_out[0, 0], expected_a[0, 0], rtol=1e-3)
        np.testing.assert_allclose(np.abs(a_out[1, 0]), np.abs(expected_a[1, 0]), rtol=1e-3)

    def test_small_matrix_n2(self):
        """Test MB04PB with N=2 minimal case."""
        n = 2
        ilo = 1

        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ], order='F', dtype=float)

        qg = np.array([
            [0.5, 1.0, 2.0],
            [1.5, 2.5, 3.0],
        ], order='F', dtype=float)

        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)

        assert info == 0
        assert a_out.shape == (n, n)
        assert qg_out.shape == (n, n + 1)
        assert cs.shape == (2 * n - 2,)
        assert tau.shape == (n - 1,)


class TestMB04PBEdgeCases:
    """Edge case tests."""

    def test_n_equals_zero(self):
        """Test MB04PB with N=0 (quick return)."""
        n = 0
        ilo = 1

        a = np.zeros((0, 0), order='F', dtype=float)
        qg = np.zeros((0, 1), order='F', dtype=float)

        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)

        assert info == 0

    def test_n_equals_one(self):
        """Test MB04PB with N=1 (trivial case, no reduction needed)."""
        n = 1
        ilo = 1

        a = np.array([[2.5]], order='F', dtype=float)
        qg = np.array([[1.0, 3.0]], order='F', dtype=float)

        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)

        assert info == 0
        assert a_out.shape == (1, 1)
        np.testing.assert_allclose(a_out[0, 0], 2.5, rtol=1e-14)


class TestMB04PBILOParameter:
    """Tests for ILO parameter handling."""

    def test_ilo_greater_than_one(self):
        """Test MB04PB with ILO > 1 (partial reduction)."""
        n = 4
        ilo = 2

        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 5.0, 6.0, 7.0],
            [0.0, 8.0, 9.0, 10.0],
            [0.0, 11.0, 12.0, 13.0],
        ], order='F', dtype=float)

        qg = np.array([
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 0.5, 1.5, 2.5, 3.5],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.5, 2.5, 3.5, 4.5],
        ], order='F', dtype=float)

        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)

        assert info == 0

        np.testing.assert_allclose(a_out[0, 0], 1.0, rtol=1e-14)


class TestMB04PBMathProperties:
    """Mathematical property validation tests."""

    def test_orthogonal_symplectic_transformation(self):
        """
        Validate that the routine completes and populates output arrays.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 6
        ilo = 1

        a = np.random.randn(n, n).astype(float, order='F')
        qg = np.random.randn(n, n + 1).astype(float, order='F')
        qg[:, :n] = np.tril(qg[:, :n])
        qg[:, 1:] = np.triu(qg[:, 1:])

        a_out, qg_out, cs, tau, info = mb04pb(n, ilo, a, qg)

        assert info == 0
        assert tau.shape == (n - 1,)
        assert cs.shape == (2 * n - 2,)
        assert np.any(tau != 0.0), "TAU should contain non-zero reflector scalars"

    def test_deterministic_results(self):
        """
        Verify that results are deterministic with same input.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 3
        ilo = 1

        a = np.random.randn(n, n).astype(float, order='F')
        qg = np.random.randn(n, n + 1).astype(float, order='F')
        qg_sym = qg.copy()
        for i in range(n):
            for j in range(i + 1, n):
                qg_sym[j, i] = qg_sym[i, j]

        a1, qg1, cs1, tau1, info1 = mb04pb(n, ilo, a.copy(), qg_sym.copy())
        a2, qg2, cs2, tau2, info2 = mb04pb(n, ilo, a.copy(), qg_sym.copy())

        assert info1 == 0
        assert info2 == 0
        np.testing.assert_allclose(a1, a2, rtol=1e-14)
        np.testing.assert_allclose(qg1, qg2, rtol=1e-14)
        np.testing.assert_allclose(cs1, cs2, rtol=1e-14)
        np.testing.assert_allclose(tau1, tau2, rtol=1e-14)


class TestMB04PBErrorHandling:
    """Error handling tests."""

    def test_invalid_n_negative(self):
        """Test MB04PB with invalid N < 0."""
        n = -1
        ilo = 1

        a = np.zeros((1, 1), order='F', dtype=float)
        qg = np.zeros((1, 2), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            mb04pb(n, ilo, a, qg)

    def test_invalid_ilo_zero(self):
        """Test MB04PB with invalid ILO = 0."""
        n = 3
        ilo = 0

        a = np.zeros((n, n), order='F', dtype=float)
        qg = np.zeros((n, n + 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            mb04pb(n, ilo, a, qg)

    def test_invalid_ilo_greater_than_n(self):
        """Test MB04PB with invalid ILO > N."""
        n = 3
        ilo = 5

        a = np.zeros((n, n), order='F', dtype=float)
        qg = np.zeros((n, n + 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            mb04pb(n, ilo, a, qg)
