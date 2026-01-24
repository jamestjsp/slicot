// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>

void mb04zd(const char *compu, i32 n, f64 *a, i32 lda, f64 *qg, i32 ldqg,
            f64 *u, i32 ldu, f64 *dwork, i32 *info) {
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;
    const i32 inc1 = 1;

    bool accum = (*compu == 'A' || *compu == 'a' ||
                  *compu == 'V' || *compu == 'v');
    bool form = (*compu == 'F' || *compu == 'f' ||
                 *compu == 'I' || *compu == 'i');
    bool forget = (*compu == 'N' || *compu == 'n');

    *info = 0;

    if (!accum && !form && !forget) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ldqg < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldu < 1 || (!forget && ldu < (n > 1 ? n : 1))) {
        *info = -8;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    f64 t[4];

    // Loop: Fortran J=1..N-1 corresponds to C j=0..n-2
    // In Fortran terms: J = j+1, N-J = n-j-1
    for (i32 j = 0; j < n - 1; j++) {
        i32 J = j + 1;        // Fortran J (1-indexed)
        i32 nmj = n - J;      // N-J in Fortran = n - j - 1

        // Copy row J of Q to DWORK(N+1:2N)
        // Fortran: DCOPY(J-1, QG(J,1), LDQG, DWORK(N+1), 1)
        // Copies J-1 elements from QG(J,1:J-1) to DWORK(N+1:N+J-1)
        if (J > 1) {
            i32 jm1 = J - 1;
            SLC_DCOPY(&jm1, &qg[j], &ldqg, &dwork[n], &inc1);
        }

        // Fortran: DCOPY(N-J+1, QG(J,J), 1, DWORK(N+J), 1)
        // Copies N-J+1 elements from QG(J,J:N) to DWORK(N+J:2N)
        i32 nmjp1 = nmj + 1;
        SLC_DCOPY(&nmjp1, &qg[j + j*ldqg], &inc1, &dwork[n + j], &inc1);

        // DWORK(J+1:N) = -A(:,J+1:N)^T * DWORK(N+1:2N)
        // Fortran: DGEMV('T', N, N-J, -ONE, A(1,J+1), LDA, DWORK(N+1), 1, ZERO, DWORK(J+1), 1)
        f64 neg_one = -one;
        if (nmj > 0) {
            SLC_DGEMV("T", &n, &nmj, &neg_one, &a[0 + J*lda], &lda,
                      &dwork[n], &inc1, &zero, &dwork[J], &inc1);
        }

        // DWORK(J+1:N) += Q(J+1:N, 1:J) * A(1:J, J)
        // Fortran: DGEMV('N', N-J, J, ONE, QG(J+1,1), LDQG, A(1,J), 1, ONE, DWORK(J+1), 1)
        if (nmj > 0) {
            SLC_DGEMV("N", &nmj, &J, &one, &qg[J], &ldqg,
                      &a[0 + j*lda], &inc1, &one, &dwork[J], &inc1);
        }

        // DWORK(J+1:N) += Q_sym(J+1:N, J+1:N) * A(J+1:N, J)
        // Fortran: DSYMV('L', N-J, ONE, QG(J+1,J+1), LDQG, A(J+1,J), 1, ONE, DWORK(J+1), 1)
        if (nmj > 0) {
            SLC_DSYMV("L", &nmj, &one, &qg[J + J*ldqg], &ldqg,
                      &a[J + j*lda], &inc1, &one, &dwork[J], &inc1);
        }

        // First symplectic reflection
        // Fortran: DLARFG(N-J, DWORK(J+1), DWORK(J+2), 1, TAU)
        f64 tau;
        if (nmj > 0) {
            SLC_DLARFG(&nmj, &dwork[J], &dwork[J+1], &inc1, &tau);
        } else {
            tau = zero;
        }
        f64 y = dwork[J];
        dwork[J] = one;

        // Apply reflector from left to A(J+1:N, 1:N)
        // Fortran: DLARFX('L', N-J, N, DWORK(J+1), TAU, A(J+1,1), LDA, DWORK(N+1))
        if (nmj > 0) {
            SLC_DLARFX("L", &nmj, &n, &dwork[J], &tau,
                       &a[J + 0*lda], &lda, &dwork[n]);
        }

        // Apply reflector from right to A(1:N, J+1:N)
        // Fortran: DLARFX('R', N, N-J, DWORK(J+1), TAU, A(1,J+1), LDA, DWORK(N+1))
        if (nmj > 0) {
            SLC_DLARFX("R", &n, &nmj, &dwork[J], &tau,
                       &a[0 + J*lda], &lda, &dwork[n]);
        }

        // Apply reflector from left to Q(J+1:N, 1:J)
        // Fortran: DLARFX('L', N-J, J, DWORK(J+1), TAU, QG(J+1,1), LDQG, DWORK(N+1))
        if (nmj > 0 && J > 0) {
            SLC_DLARFX("L", &nmj, &J, &dwork[J], &tau,
                       &qg[J], &ldqg, &dwork[n]);
        }

        // Symmetric update to Q(J+1:N, J+1:N)
        if (nmj > 0) {
            SLC_DSYMV("L", &nmj, &tau, &qg[J + J*ldqg], &ldqg,
                      &dwork[J], &inc1, &zero, &dwork[n+J], &inc1);
            f64 dot_val = SLC_DDOT(&nmj, &dwork[n+J], &inc1, &dwork[J], &inc1);
            f64 factor = -tau * dot_val / two;
            SLC_DAXPY(&nmj, &factor, &dwork[J], &inc1, &dwork[n+J], &inc1);
            neg_one = -one;
            SLC_DSYR2("L", &nmj, &neg_one, &dwork[J], &inc1,
                      &dwork[n+J], &inc1, &qg[J + J*ldqg], &ldqg);
        }

        // Apply reflector from right to G(1:J, J+1:N)
        // Fortran: DLARFX('R', J, N-J, DWORK(J+1), TAU, QG(1,J+2), LDQG, DWORK(N+1))
        if (nmj > 0 && J > 0) {
            SLC_DLARFX("R", &J, &nmj, &dwork[J], &tau,
                       &qg[0 + (J+1)*ldqg], &ldqg, &dwork[n]);
        }

        // Symmetric update to G(J+1:N, J+1:N)
        if (nmj > 0) {
            SLC_DSYMV("U", &nmj, &tau, &qg[J + (J+1)*ldqg], &ldqg,
                      &dwork[J], &inc1, &zero, &dwork[n+J], &inc1);
            f64 dot_val = SLC_DDOT(&nmj, &dwork[n+J], &inc1, &dwork[J], &inc1);
            f64 factor = -tau * dot_val / two;
            SLC_DAXPY(&nmj, &factor, &dwork[J], &inc1, &dwork[n+J], &inc1);
            neg_one = -one;
            SLC_DSYR2("U", &nmj, &neg_one, &dwork[J], &inc1,
                      &dwork[n+J], &inc1, &qg[J + (J+1)*ldqg], &ldqg);
        }

        if (form) {
            // Save reflection: U(J+1:N, J) = DWORK(J+1:N), U(J+1,J) = TAU
            // Fortran: DCOPY(N-J, DWORK(J+1), 1, U(J+1,J), 1)
            if (nmj > 0) {
                SLC_DCOPY(&nmj, &dwork[J], &inc1, &u[J + j*ldu], &inc1);
            }
            u[J + j*ldu] = tau;
        } else if (accum) {
            // Accumulate reflection
            if (nmj > 0) {
                SLC_DLARFX("R", &n, &nmj, &dwork[J], &tau,
                           &u[0 + J*ldu], &ldu, &dwork[n]);
                SLC_DLARFX("R", &n, &nmj, &dwork[J], &tau,
                           &u[0 + (n+J)*ldu], &ldu, &dwork[n]);
            }
        }

        // X = G(:,J+1)' * Q(:,J) + A(J+1,:) * A(:,J)
        // Fortran: X = DDOT(J, QG(1,J+2), 1, QG(J,1), LDQG) +
        //              DDOT(N-J, QG(J+1,J+2), LDQG, QG(J+1,J), 1) +
        //              DDOT(N, A(J+1,1), LDA, A(1,J), 1)
        f64 x_val = 0.0;
        if (J > 0) {
            x_val = SLC_DDOT(&J, &qg[0 + (J+1)*ldqg], &inc1, &qg[j], &ldqg);
        }
        if (nmj > 0) {
            x_val += SLC_DDOT(&nmj, &qg[J + (J+1)*ldqg], &ldqg,
                              &qg[J + j*ldqg], &inc1);
        }
        x_val += SLC_DDOT(&n, &a[J], &lda, &a[0 + j*lda], &inc1);

        // Symplectic rotation to zero (H*H)(N+J+1, J)
        f64 cosine, sine, temp;
        SLC_DLARTG(&x_val, &y, &cosine, &sine, &temp);

        // Apply rotation
        // Fortran: DROT(J, A(J+1,1), LDA, QG(J+1,1), LDQG, COSINE, SINE)
        if (J > 0) {
            SLC_DROT(&J, &a[J], &lda, &qg[J], &ldqg, &cosine, &sine);
        }
        // Fortran: DROT(J, A(1,J+1), 1, QG(1,J+2), 1, COSINE, SINE)
        if (J > 0) {
            SLC_DROT(&J, &a[0 + J*lda], &inc1, &qg[0 + (J+1)*ldqg], &inc1,
                     &cosine, &sine);
        }

        if (J < n - 1) {
            i32 nmj1 = n - J - 1;
            // Fortran: DROT(N-J-1, A(J+1,J+2), LDA, QG(J+2,J+1), 1, COSINE, SINE)
            SLC_DROT(&nmj1, &a[J + (J+1)*lda], &lda, &qg[(J+1) + J*ldqg],
                     &inc1, &cosine, &sine);
            // Fortran: DROT(N-J-1, A(J+2,J+1), 1, QG(J+1,J+3), LDQG, COSINE, SINE)
            SLC_DROT(&nmj1, &a[(J+1) + J*lda], &inc1, &qg[J + (J+2)*ldqg],
                     &ldqg, &cosine, &sine);
        }

        // 2x2 rotation on diagonal block
        t[0] = a[J + J*lda];          // T(1,1) = A(J+1,J+1)
        t[1] = qg[J + J*ldqg];        // T(2,1) = Q(J+1,J+1)
        t[2] = qg[J + (J+1)*ldqg];    // T(1,2) = G(J+1,J+1)
        t[3] = -t[0];                  // T(2,2) = -A(J+1,J+1)

        i32 two_val = 2;
        SLC_DROT(&two_val, &t[0], &inc1, &t[2], &inc1, &cosine, &sine);
        SLC_DROT(&two_val, &t[0], &two_val, &t[1], &two_val, &cosine, &sine);
        a[J + J*lda] = t[0];
        qg[J + (J+1)*ldqg] = t[2];
        qg[J + J*ldqg] = t[1];

        if (form) {
            u[j + j*ldu] = cosine;
            u[j + (n+j)*ldu] = sine;
        } else if (accum) {
            SLC_DROT(&n, &u[0 + J*ldu], &inc1, &u[0 + (n+J)*ldu], &inc1,
                     &cosine, &sine);
        }

        // Second set: DWORK := (A*A + G*Q)(J+1:N, J)
        // Fortran: DGEMV('N', N-J, N, ONE, A(J+1,1), LDA, A(1,J), 1, ZERO, DWORK(J+1), 1)
        if (nmj > 0) {
            SLC_DGEMV("N", &nmj, &n, &one, &a[J], &lda,
                      &a[0 + j*lda], &inc1, &zero, &dwork[J], &inc1);
        }
        // Fortran: DGEMV('T', J, N-J, ONE, QG(1,J+2), LDQG, QG(J,1), LDQG, ONE, DWORK(J+1), 1)
        if (nmj > 0 && J > 0) {
            SLC_DGEMV("T", &J, &nmj, &one, &qg[0 + (J+1)*ldqg], &ldqg,
                      &qg[j], &ldqg, &one, &dwork[J], &inc1);
        }
        // Fortran: DSYMV('U', N-J, ONE, QG(J+1,J+2), LDQG, QG(J+1,J), 1, ONE, DWORK(J+1), 1)
        if (nmj > 0) {
            SLC_DSYMV("U", &nmj, &one, &qg[J + (J+1)*ldqg], &ldqg,
                      &qg[J + j*ldqg], &inc1, &one, &dwork[J], &inc1);
        }

        // Second symplectic reflection
        if (nmj > 0) {
            SLC_DLARFG(&nmj, &dwork[J], &dwork[J+1], &inc1, &tau);
        } else {
            tau = zero;
        }
        dwork[J] = one;

        // Apply reflector from left to A(J+1:N, 1:N)
        if (nmj > 0) {
            SLC_DLARFX("L", &nmj, &n, &dwork[J], &tau,
                       &a[J + 0*lda], &lda, &dwork[n]);
        }

        // Apply reflector from right to A(1:N, J+1:N)
        if (nmj > 0) {
            SLC_DLARFX("R", &n, &nmj, &dwork[J], &tau,
                       &a[0 + J*lda], &lda, &dwork[n]);
        }

        // Apply reflector from left to Q(J+1:N, 1:J)
        if (nmj > 0 && J > 0) {
            SLC_DLARFX("L", &nmj, &J, &dwork[J], &tau,
                       &qg[J], &ldqg, &dwork[n]);
        }

        // Symmetric update to Q(J+1:N, J+1:N)
        if (nmj > 0) {
            SLC_DSYMV("L", &nmj, &tau, &qg[J + J*ldqg], &ldqg,
                      &dwork[J], &inc1, &zero, &dwork[n+J], &inc1);
            f64 dot_val = SLC_DDOT(&nmj, &dwork[n+J], &inc1, &dwork[J], &inc1);
            f64 factor = -tau * dot_val / two;
            SLC_DAXPY(&nmj, &factor, &dwork[J], &inc1, &dwork[n+J], &inc1);
            neg_one = -one;
            SLC_DSYR2("L", &nmj, &neg_one, &dwork[J], &inc1,
                      &dwork[n+J], &inc1, &qg[J + J*ldqg], &ldqg);
        }

        // Apply reflector from right to G(1:J, J+1:N)
        if (nmj > 0 && J > 0) {
            SLC_DLARFX("R", &J, &nmj, &dwork[J], &tau,
                       &qg[0 + (J+1)*ldqg], &ldqg, &dwork[n]);
        }

        // Symmetric update to G(J+1:N, J+1:N)
        if (nmj > 0) {
            SLC_DSYMV("U", &nmj, &tau, &qg[J + (J+1)*ldqg], &ldqg,
                      &dwork[J], &inc1, &zero, &dwork[n+J], &inc1);
            f64 dot_val = SLC_DDOT(&nmj, &dwork[n+J], &inc1, &dwork[J], &inc1);
            f64 factor = -tau * dot_val / two;
            SLC_DAXPY(&nmj, &factor, &dwork[J], &inc1, &dwork[n+J], &inc1);
            neg_one = -one;
            SLC_DSYR2("U", &nmj, &neg_one, &dwork[J], &inc1,
                      &dwork[n+J], &inc1, &qg[J + (J+1)*ldqg], &ldqg);
        }

        if (form) {
            if (nmj > 0) {
                SLC_DCOPY(&nmj, &dwork[J], &inc1, &u[J + (n+j)*ldu], &inc1);
            }
            u[J + (n+j)*ldu] = tau;
        } else if (accum) {
            if (nmj > 0) {
                SLC_DLARFX("R", &n, &nmj, &dwork[J], &tau,
                           &u[0 + J*ldu], &ldu, &dwork[n]);
                SLC_DLARFX("R", &n, &nmj, &dwork[J], &tau,
                           &u[0 + (n+J)*ldu], &ldu, &dwork[n]);
            }
        }
    }

    if (form) {
        // Form S by accumulating transformations (backward)
        for (i32 j = n - 2; j >= 0; j--) {
            i32 J = j + 1;
            i32 nmj = n - J;

            // Initialize (J+1)st column of S
            for (i32 i = 0; i < n; i++) {
                u[i + J*ldu] = zero;
                u[i + (n+J)*ldu] = zero;
            }
            u[J + J*ldu] = one;

            // Second reflection
            f64 tau = u[J + (n+j)*ldu];
            u[J + (n+j)*ldu] = one;
            if (nmj > 0) {
                SLC_DLARFX("L", &nmj, &nmj, &u[J + (n+j)*ldu], &tau,
                           &u[J + J*ldu], &ldu, &dwork[n]);
                SLC_DLARFX("L", &nmj, &nmj, &u[J + (n+j)*ldu], &tau,
                           &u[J + (n+J)*ldu], &ldu, &dwork[n]);
            }

            // Rotation
            if (nmj > 0) {
                SLC_DROT(&nmj, &u[J + J*ldu], &ldu, &u[J + (n+J)*ldu],
                         &ldu, &u[j + j*ldu], &u[j + (n+j)*ldu]);
            }

            // First reflection
            tau = u[J + j*ldu];
            u[J + j*ldu] = one;
            if (nmj > 0) {
                SLC_DLARFX("L", &nmj, &nmj, &u[J + j*ldu], &tau,
                           &u[J + J*ldu], &ldu, &dwork[n]);
                SLC_DLARFX("L", &nmj, &nmj, &u[J + j*ldu], &tau,
                           &u[J + (n+J)*ldu], &ldu, &dwork[n]);
            }
        }

        // First column is first column of identity
        for (i32 i = 0; i < n; i++) {
            u[i] = zero;
            u[i + n*ldu] = zero;
        }
        u[0] = one;
    }
}
