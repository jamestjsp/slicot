"""
Tests for SB10VD - State feedback and output injection for H2 controller.
"""
import numpy as np
import pytest
import slicot


class TestSB10VDBasic:
    """Basic functionality tests for SB10VD."""

    def test_basic(self):
        """Test basic SB10VD functionality."""
        n, m, np_, ncon, nmeas = 6, 5, 5, 2, 2

        a = np.array([
            [-1.0, -2.0, -6.0, -8.0,  2.0,  3.0],
            [ 0.0,  4.0,  9.0,  4.0,  5.0, -5.0],
            [ 4.0, -7.0, -5.0,  7.0,  8.0,  8.0],
            [ 5.0, -2.0,  0.0, -1.0, -9.0,  0.0],
            [-3.0,  0.0,  2.0, -3.0,  1.0,  2.0],
            [-2.0,  3.0, -1.0,  0.0, -4.0, -6.0]
        ], dtype=np.float64, order='F')

        b = np.array([
            [-3.0,  2.0, -5.0,  4.0, -3.0,  1.0],
            [-4.0,  0.0, -7.0, -6.0,  9.0, -2.0],
            [-2.0,  1.0,  0.0,  1.0, -8.0,  3.0],
            [ 1.0, -5.0,  7.0,  1.0,  0.0, -6.0],
            [ 0.0,  2.0, -2.0, -2.0,  5.0, -2.0]
        ], dtype=np.float64, order='F').T

        c = np.array([
            [ 1.0, -3.0, -7.0,  9.0,  0.0],
            [-1.0,  0.0,  5.0, -3.0,  1.0],
            [ 2.0,  5.0,  0.0,  4.0, -2.0],
            [-4.0, -1.0, -8.0,  0.0,  1.0],
            [ 0.0,  1.0,  2.0,  3.0, -6.0],
            [-3.0,  1.0, -2.0,  7.0, -2.0]
        ], dtype=np.float64, order='F').T

        f, h, x, y, xcond, ycond, info = slicot.sb10vd(ncon, nmeas, a, b, c)

        assert info == 0, f"SB10VD returned info = {info}"
        assert f.shape == (ncon, n)
        assert h.shape == (n, nmeas)
        assert x.shape == (n, n)
        assert y.shape == (n, n)
        assert xcond >= 0
        assert ycond >= 0


class TestSB10VDEdgeCases:
    """Edge case tests for SB10VD."""

    def test_small_system(self):
        """Test with a 2x2 stable system."""
        n, m, np_, ncon, nmeas = 2, 2, 2, 1, 1

        a = np.array([[-1.0, 0.5], [0.0, -2.0]], dtype=np.float64, order='F')
        b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64, order='F')
        c = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64, order='F')

        f, h, x, y, xcond, ycond, info = slicot.sb10vd(ncon, nmeas, a, b, c)

        assert info == 0, f"SB10VD returned info = {info}"
        assert f.shape == (ncon, n)
        assert h.shape == (n, nmeas)
