"""
Tests for SB02RD - Riccati Equation Solver.

Solves continuous-time algebraic Riccati equation:
    Q + op(A)'*X + X*op(A) - X*G*X = 0                         (DICO='C')

or discrete-time algebraic Riccati equation:
    Q + op(A)'*X*(I_n + G*X)^(-1)*op(A) - X = 0               (DICO='D')

where op(A) = A or A', and G is symmetric.
"""

import numpy as np
import pytest


def test_sb02rd_html_doc_example():
    """
    Validate SB02RD using SLICOT HTML documentation example.

    From SB02RD.html:
    - N=2, JOB='X', DICO='C', continuous-time Riccati
    - A = [[0, 1], [0, 0]]
    - Q = [[1, 0], [0, 2]]
    - G = [[0, 0], [0, 1]]
    - Expected X = [[2, 1], [1, 2]]

    This is the primary validation test from authoritative source.
    Note: JOB='X' is used (solution only) since condition number routines
    (SB02QD/SB02SD) are not yet implemented.
    """
    from slicot import sb02rd

    n = 2

    A = np.array([[0.0, 1.0],
                  [0.0, 0.0]], order='F', dtype=float)

    Q = np.array([[1.0, 0.0],
                  [0.0, 2.0]], order='F', dtype=float)

    G = np.array([[0.0, 0.0],
                  [0.0, 1.0]], order='F', dtype=float)

    X_expected = np.array([[2.0, 1.0],
                           [1.0, 2.0]], order='F', dtype=float)

    X, sep, rcond, ferr, wr, wi, s, info = sb02rd(
        job='X',
        dico='C',
        hinv='D',
        trana='N',
        uplo='U',
        scal='N',
        sort='S',
        fact='N',
        lyapun='O',
        A=A,
        Q=Q,
        G=G
    )

    assert info == 0, f"SB02RD failed with info={info}"

    np.testing.assert_allclose(X, X_expected, rtol=1e-3, atol=1e-4)


def test_sb02rd_continuous_basic():
    """
    Test SB02RD continuous-time case with JOB='X' (solution only).

    Random seed: 42 (for reproducibility)
    Uses stable A matrix to ensure solution exists.
    """
    from slicot import sb02rd

    np.random.seed(42)
    n = 3

    A_raw = np.random.randn(n, n)
    A = A_raw - 2.0 * np.eye(n)
    A = A.astype(float, order='F')

    Q = np.eye(n, dtype=float, order='F')

    G = np.eye(n, dtype=float, order='F') * 0.5

    X, sep, rcond, ferr, wr, wi, s, info = sb02rd(
        job='X',
        dico='C',
        hinv='D',
        trana='N',
        uplo='U',
        scal='N',
        sort='S',
        fact='N',
        lyapun='O',
        A=A,
        Q=Q,
        G=G
    )

    assert info == 0, f"SB02RD failed with info={info}"

    np.testing.assert_allclose(X, X.T, rtol=1e-14, atol=1e-15,
                               err_msg="Solution X is not symmetric")

    residual = Q + A.T @ X + X @ A - X @ G @ X
    np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-10,
                               err_msg="Riccati equation residual too large")


def test_sb02rd_discrete_basic():
    """
    Test SB02RD discrete-time case with JOB='X' (solution only).

    Random seed: 123 (for reproducibility)
    Uses stable A matrix (eigenvalues inside unit circle).
    Note: SORT='S' selects stable eigenvalues first (inside unit circle).
    """
    from slicot import sb02rd

    np.random.seed(123)
    n = 3

    A_raw = np.random.randn(n, n)
    A = A_raw * 0.3
    A = A.astype(float, order='F')

    Q = np.eye(n, dtype=float, order='F')

    G = np.eye(n, dtype=float, order='F') * 0.1

    X, sep, rcond, ferr, wr, wi, s, info = sb02rd(
        job='X',
        dico='D',
        hinv='D',
        trana='N',
        uplo='U',
        scal='N',
        sort='S',
        fact='N',
        lyapun='O',
        A=A,
        Q=Q,
        G=G
    )

    assert info == 0, f"SB02RD failed with info={info}"

    np.testing.assert_allclose(X, X.T, rtol=1e-14, atol=1e-15,
                               err_msg="Solution X is not symmetric")

    I_n = np.eye(n)
    inv_term = np.linalg.inv(I_n + G @ X)
    residual = Q + A.T @ X @ inv_term @ A - X
    np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-10,
                               err_msg="Discrete Riccati residual too large")


def test_sb02rd_transpose_form():
    """
    Test SB02RD with TRANA='T' (transpose form).

    Solves: Q + op(A)*X + X*op(A)' - X*G*X = 0 where op(A) = A'

    Random seed: 456 (for reproducibility)
    """
    from slicot import sb02rd

    np.random.seed(456)
    n = 2

    A_raw = np.random.randn(n, n)
    A = A_raw - 2.0 * np.eye(n)
    A = A.astype(float, order='F')

    Q = np.eye(n, dtype=float, order='F')
    G = np.eye(n, dtype=float, order='F') * 0.5

    X, sep, rcond, ferr, wr, wi, s, info = sb02rd(
        job='X',
        dico='C',
        hinv='D',
        trana='T',
        uplo='U',
        scal='N',
        sort='S',
        fact='N',
        lyapun='O',
        A=A,
        Q=Q,
        G=G
    )

    assert info == 0, f"SB02RD failed with info={info}"

    residual = Q + A @ X + X @ A.T - X @ G @ X
    np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-10,
                               err_msg="Transpose form residual too large")


def test_sb02rd_lower_triangle():
    """
    Test SB02RD with UPLO='L' (lower triangle storage).

    Random seed: 789 (for reproducibility)
    """
    from slicot import sb02rd

    np.random.seed(789)
    n = 3

    A_raw = np.random.randn(n, n)
    A = A_raw - 2.0 * np.eye(n)
    A = A.astype(float, order='F')

    Q = np.eye(n, dtype=float, order='F')
    G = np.eye(n, dtype=float, order='F') * 0.5

    X, sep, rcond, ferr, wr, wi, s, info = sb02rd(
        job='X',
        dico='C',
        hinv='D',
        trana='N',
        uplo='L',
        scal='N',
        sort='S',
        fact='N',
        lyapun='O',
        A=A,
        Q=Q,
        G=G
    )

    assert info == 0, f"SB02RD failed with info={info}"

    residual = Q + A.T @ X + X @ A - X @ G @ X
    np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_sb02rd_scaling():
    """
    Test SB02RD with SCAL='G' (general scaling).

    Random seed: 111 (for reproducibility)
    """
    from slicot import sb02rd

    np.random.seed(111)
    n = 3

    A_raw = np.random.randn(n, n)
    A = A_raw - 2.0 * np.eye(n)
    A = A.astype(float, order='F')

    Q = np.eye(n, dtype=float, order='F') * 100.0
    G = np.eye(n, dtype=float, order='F') * 0.01

    X, sep, rcond, ferr, wr, wi, s, info = sb02rd(
        job='X',
        dico='C',
        hinv='D',
        trana='N',
        uplo='U',
        scal='G',
        sort='S',
        fact='N',
        lyapun='O',
        A=A,
        Q=Q,
        G=G
    )

    assert info == 0, f"SB02RD failed with info={info}"

    residual = Q + A.T @ X + X @ A - X @ G @ X
    np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-6)


def test_sb02rd_n1_minimal():
    """
    Test SB02RD with N=1 (minimal case).

    For N=1, continuous-time:
    Q + A*X + X*A - X*G*X = 0

    With A=-1, Q=1, G=1:
    1 - 2*X - X^2 = 0 => X = -1 + sqrt(2) ~ 0.414
    """
    from slicot import sb02rd

    A = np.array([[-1.0]], dtype=float, order='F')
    Q = np.array([[1.0]], dtype=float, order='F')
    G = np.array([[1.0]], dtype=float, order='F')

    X, sep, rcond, ferr, wr, wi, s, info = sb02rd(
        job='X',
        dico='C',
        hinv='D',
        trana='N',
        uplo='U',
        scal='N',
        sort='S',
        fact='N',
        lyapun='O',
        A=A,
        Q=Q,
        G=G
    )

    assert info == 0
    X_expected = -1.0 + np.sqrt(2.0)
    np.testing.assert_allclose(X[0, 0], X_expected, rtol=1e-10)


def test_sb02rd_eigenvalue_preservation():
    """
    Validate mathematical property: closed-loop eigenvalues are stable.

    For continuous-time: eigenvalues of (A - G*X) have negative real parts.
    Random seed: 222 (for reproducibility)
    """
    from slicot import sb02rd

    np.random.seed(222)
    n = 4

    A_raw = np.random.randn(n, n)
    A = A_raw - 1.5 * np.eye(n)
    A = A.astype(float, order='F')

    Q = np.eye(n, dtype=float, order='F')
    G = np.eye(n, dtype=float, order='F') * 0.5

    X, sep, rcond, ferr, wr, wi, s, info = sb02rd(
        job='X',
        dico='C',
        hinv='D',
        trana='N',
        uplo='U',
        scal='N',
        sort='S',
        fact='N',
        lyapun='O',
        A=A,
        Q=Q,
        G=G
    )

    assert info == 0

    Ac = A - G @ X
    eigs = np.linalg.eigvals(Ac)

    assert np.all(eigs.real < 0), \
        f"Closed-loop eigenvalues not all stable: {eigs}"


def test_sb02rd_positive_semidefinite_solution():
    """
    Validate mathematical property: X is positive semidefinite.

    For stabilizable/detectable systems, unique X >= 0 exists.
    Random seed: 333 (for reproducibility)
    """
    from slicot import sb02rd

    np.random.seed(333)
    n = 3

    A_raw = np.random.randn(n, n)
    A = A_raw - 2.0 * np.eye(n)
    A = A.astype(float, order='F')

    Q = np.eye(n, dtype=float, order='F')
    G = np.eye(n, dtype=float, order='F') * 0.5

    X, sep, rcond, ferr, wr, wi, s, info = sb02rd(
        job='X',
        dico='C',
        hinv='D',
        trana='N',
        uplo='U',
        scal='N',
        sort='S',
        fact='N',
        lyapun='O',
        A=A,
        Q=Q,
        G=G
    )

    assert info == 0

    eigvals = np.linalg.eigvalsh(X)
    assert np.all(eigvals >= -1e-12), \
        f"X is not positive semidefinite: eigenvalues={eigvals}"


def test_sb02rd_error_invalid_job():
    """
    Test SB02RD error handling: invalid JOB parameter.
    """
    from slicot import sb02rd

    A = np.eye(2, dtype=float, order='F')
    Q = np.eye(2, dtype=float, order='F')
    G = np.eye(2, dtype=float, order='F')

    with pytest.raises(ValueError, match="[Jj][Oo][Bb]|[Pp]arameter"):
        sb02rd(
            job='Z',
            dico='C',
            hinv='D',
            trana='N',
            uplo='U',
            scal='N',
            sort='S',
            fact='N',
            lyapun='O',
            A=A,
            Q=Q,
            G=G
        )


def test_sb02rd_error_invalid_dico():
    """
    Test SB02RD error handling: invalid DICO parameter.
    """
    from slicot import sb02rd

    A = np.eye(2, dtype=float, order='F')
    Q = np.eye(2, dtype=float, order='F')
    G = np.eye(2, dtype=float, order='F')

    with pytest.raises(ValueError, match="[Dd][Ii][Cc][Oo]|[Pp]arameter"):
        sb02rd(
            job='X',
            dico='Z',
            hinv='D',
            trana='N',
            uplo='U',
            scal='N',
            sort='S',
            fact='N',
            lyapun='O',
            A=A,
            Q=Q,
            G=G
        )


# @pytest.mark.skip(reason="SLICOT SB02RU bug: HINV='I' S12 transpose loop double-processes off-diagonals")
def test_sb02rd_discrete_hinv_inverse():
    """
    Test SB02RD discrete-time with HINV='I' (inverse symplectic).

    Random seed: 444 (for reproducibility)
    Note: For HINV='I', we need SORT='U' (unstable eigenvalues first)
    to select eigenvalues outside the unit circle.
    """
    from slicot import sb02rd

    np.random.seed(444)
    n = 3

    A_raw = np.random.randn(n, n)
    A = A_raw * 0.3
    A = A.astype(float, order='F')

    Q = np.eye(n, dtype=float, order='F')
    G = np.eye(n, dtype=float, order='F') * 0.1

    X, sep, rcond, ferr, wr, wi, s, info = sb02rd(
        job='X',
        dico='D',
        hinv='I',
        trana='N',
        uplo='U',
        scal='N',
        sort='U',
        fact='N',
        lyapun='O',
        A=A,
        Q=Q,
        G=G
    )

    assert info == 0

    I_n = np.eye(n)
    inv_term = np.linalg.inv(I_n + G @ X)
    residual = Q + A.T @ X @ inv_term @ A - X
    np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-10)
