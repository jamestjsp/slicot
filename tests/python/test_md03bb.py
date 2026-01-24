import unittest
import numpy as np
from slicot import md03bb

def test_md03bb_basic():
    """
    Validate MD03BB via wrapper.
    Should call MD03BY.
    """
    np.random.seed(42)
    n = 3
    # IPAR is unused but required
    ipar = np.array([0], dtype=np.int32)
    
    # R upper triangular (well-conditioned)
    r = np.triu(np.random.rand(n, n))
    for i in range(n): r[i, i] += 2.0
    r_in = np.asfortranarray(r)
    
    ipvt = np.arange(1, n+1, dtype=np.int32)
    diag = np.ones(n)
    qtb = np.random.rand(n)
    delta = 1.0
    par = 0.0
    ranks = np.array([n], dtype=np.int32)
    tol = 0.0
    
    # Call md03bb
    # r_out, par_out, ranks_out, x, rx, info = md03bb(cond, n, ipar, r, ipvt, diag, qtb, delta, par, ranks, tol)
    
    r_out, par_out, ranks_out, x, rx, info = md03bb('N', n, ipar, r_in, ipvt, diag, qtb, delta, par, ranks, tol)
    
    assert info == 0
    assert par_out >= 0.0
    assert x.shape == (n,)
    assert rx.shape == (n,)
    # ranks_out is array(1)
    assert ranks_out[0] <= n

if __name__ == '__main__':
    unittest.main()
