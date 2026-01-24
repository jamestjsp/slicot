/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

/**
 * @brief Single-input state feedback matrix for pole assignment.
 *
 * Determines the one-dimensional state feedback matrix G of the
 * linear time-invariant single-input system
 *
 *       dX/dt = A * X + B * U,
 *
 * where A is an NCONT-by-NCONT matrix and B is an NCONT element
 * vector such that the closed-loop system
 *
 *       dX/dt = (A - B * G) * X
 *
 * has desired poles. The system must be preliminarily reduced
 * to orthogonal canonical form using AB01MD.
 *
 * @param[in] ncont Controllable order from AB01MD (ncont >= 0)
 * @param[in] n Order of matrix Z (n >= ncont)
 * @param[in,out] a On entry: canonical form of A from AB01MD.
 *                  On exit: upper quasi-triangular Schur form S of (A-B*G)
 * @param[in] lda Leading dimension of A (lda >= max(1, ncont))
 * @param[in,out] b On entry: canonical form of B from AB01MD.
 *                  On exit: transformed vector Z * B
 * @param[in] wr Real parts of desired closed-loop poles
 * @param[in] wi Imaginary parts of desired closed-loop poles
 *               (complex conjugate pairs must appear consecutively)
 * @param[in,out] z On entry: orthogonal transformation from AB01MD.
 *                  On exit: orthogonal matrix reducing (A-B*G) to Schur form
 * @param[in] ldz Leading dimension of Z (ldz >= max(1, n))
 * @param[out] g State feedback matrix (ncont elements)
 * @param[out] dwork Workspace of size 3*ncont
 * @param[out] info Exit status: 0=success, <0=parameter -info invalid
 */
void sb01md(
    const i32 ncont,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* b,
    const f64* wr,
    const f64* wi,
    f64* z,
    const i32 ldz,
    f64* g,
    f64* dwork,
    i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    *info = 0;

    // Parameter validation
    if (ncont < 0) {
        *info = -1;
    } else if (n < ncont) {
        *info = -2;
    } else if (lda < (ncont > 1 ? ncont : 1)) {
        *info = -4;
    } else if (ldz < (n > 1 ? n : 1)) {
        *info = -9;
    }

    if (*info != 0) {
        return;
    }

    // Quick return if possible
    if (ncont == 0 || n == 0) {
        return;
    }

    // Return if system is not completely controllable
    if (b[0] == ZERO) {
        return;
    }

    // 1-by-1 case
    if (ncont == 1) {
        f64 p = a[0] - wr[0];
        a[0] = wr[0];
        g[0] = p / b[0];
        z[0] = ONE;
        return;
    }

    // General case. Save WI in DWORK(2*NCONT+1:3*NCONT)
    i32 int1 = 1;
    i32 ncont2 = 2 * ncont;
    SLC_DCOPY(&ncont, wi, &int1, &dwork[ncont2], &int1);

    f64 b1 = b[0];
    b[0] = ONE;

    i32 l = 0;    // Current eigenvalue index (0-based becomes 1-based in loop)
    i32 ll = 0;   // Counter for complex conjugate pair processing
    i32 lp1 = 0;  // L+1 when L != NCONT, preserved across iterations

    while (1) {
        l++;
        ll++;

        bool compl = (dwork[ncont2 + l - 1] != ZERO);

        if (l != ncont) {
            lp1 = l + 1;  // Update lp1 only when l != ncont

            if (ll != 2) {
                if (compl) {
                    // Compute complex eigenvector
                    dwork[ncont - 1] = ONE;
                    dwork[ncont2 - 1] = ONE;
                    f64 p = wr[l - 1];
                    f64 t = dwork[ncont2 + l - 1];
                    f64 q = t * dwork[ncont2 + lp1 - 1];
                    dwork[ncont2 + l - 1] = ONE;
                    dwork[ncont2 + lp1 - 1] = q;

                    for (i32 i = ncont; i >= lp1; i--) {
                        i32 im1 = i - 1;
                        // Compute dot product: DOT(NCONT-IM1, A(I,I), LDA, DWORK(I), 1)
                        // In Fortran: A(I,I:NCONT) dot DWORK(I:NCONT)
                        // In C: a[(i-1) + (j-1)*lda] for j = i to ncont
                        f64 dot_r = 0.0;
                        f64 dot_i = 0.0;
                        for (i32 j = i; j <= ncont; j++) {
                            dot_r += a[(i - 1) + (j - 1) * lda] * dwork[j - 1];
                            dot_i += a[(i - 1) + (j - 1) * lda] * dwork[ncont + j - 1];
                        }
                        dwork[im1 - 1] = (p * dwork[i - 1] + q * dwork[ncont + i - 1] - dot_r) / a[(i - 1) + (im1 - 1) * lda];
                        dwork[ncont + im1 - 1] = (p * dwork[ncont + i - 1] + dwork[i - 1] - dot_i) / a[(i - 1) + (im1 - 1) * lda];
                    }
                } else {
                    // Compute real eigenvector
                    dwork[ncont - 1] = ONE;
                    f64 p = wr[l - 1];

                    for (i32 i = ncont; i >= lp1; i--) {
                        i32 im1 = i - 1;
                        f64 dot = 0.0;
                        for (i32 j = i; j <= ncont; j++) {
                            dot += a[(i - 1) + (j - 1) * lda] * dwork[j - 1];
                        }
                        dwork[im1 - 1] = (p * dwork[i - 1] - dot) / a[(i - 1) + (im1 - 1) * lda];
                    }
                }
            }

            // Transform eigenvector
            for (i32 k = ncont - 1; k >= l; k--) {
                f64 r, s;
                if (ll != 2) {
                    r = dwork[k - 1];
                    s = dwork[k];
                } else {
                    r = dwork[ncont + k - 1];
                    s = dwork[ncont + k];
                }

                f64 p, q, t_val;
                SLC_DLARTG(&r, &s, &p, &q, &t_val);
                dwork[k - 1] = t_val;

                i32 nj;
                if (ll != 2) {
                    nj = (k - 1 > l) ? k - 1 : l;  // MAX(K-1, L)
                } else {
                    dwork[ncont + k - 1] = t_val;
                    nj = l - 1;
                }

                // Transform A: row rotation
                // CALL DROT(NCONT-NJ+1, A(K,NJ), LDA, A(K+1,NJ), LDA, P, Q)
                i32 rot_n = ncont - nj + 1;
                SLC_DROT(&rot_n, &a[(k - 1) + (nj - 1) * lda], &lda, &a[k + (nj - 1) * lda], &lda, &p, &q);

                i32 ni;
                if (compl && ll == 1) {
                    ni = ncont;
                } else {
                    ni = (k + 2 < ncont) ? k + 2 : ncont;  // MIN(K+2, NCONT)
                }

                // Transform A: column rotation
                // CALL DROT(NI, A(1,K), 1, A(1,K+1), 1, P, Q)
                SLC_DROT(&ni, &a[(k - 1) * lda], &int1, &a[k * lda], &int1, &p, &q);

                if (k == l) {
                    // Transform B
                    t_val = b[k - 1];
                    b[k - 1] = p * t_val;
                    b[k] = -q * t_val;
                }

                // Accumulate transformations in Z
                // CALL DROT(NCONT, Z(1,K), 1, Z(1,K+1), 1, P, Q)
                SLC_DROT(&ncont, &z[(k - 1) * ldz], &int1, &z[k * ldz], &int1, &p, &q);

                if (compl && ll != 2) {
                    t_val = dwork[ncont + k - 1];
                    dwork[ncont + k - 1] = p * t_val + q * dwork[ncont + k];
                    dwork[ncont + k] = p * dwork[ncont + k] - q * t_val;
                }
            }
        }

        if (!compl) {
            // Find one element of G (real eigenvalue)
            i32 k = l;
            f64 r = b[l - 1];
            if (l != ncont) {
                if (fabs(b[lp1 - 1]) > fabs(b[l - 1])) {
                    k = lp1;
                    r = b[lp1 - 1];
                }
            }

            f64 p = a[(k - 1) + (l - 1) * lda];
            if (k == l) {
                p = p - wr[l - 1];
            }
            p = p / r;

            // lp1 holds correct value from last l != ncont iteration
            i32 lp1_val = (l != ncont) ? l + 1 : lp1;
            f64 mp = -p;
            SLC_DAXPY(&lp1_val, &mp, b, &int1, &a[(l - 1) * lda], &int1);

            g[l - 1] = p / b1;

            if (l != ncont) {
                ll = 0;
                continue;  // GO TO 20
            }
        } else if (ll == 1) {
            continue;  // GO TO 20
        } else {
            // Find two elements of G (complex conjugate pair)
            i32 k = l;
            f64 r = b[l - 1];
            if (l != ncont) {
                if (fabs(b[lp1 - 1]) > fabs(b[l - 1])) {
                    k = lp1;
                    r = b[lp1 - 1];
                }
            }

            f64 p = a[(k - 1) + (l - 2) * lda];  // A(K, L-1)
            f64 q = a[(k - 1) + (l - 1) * lda];  // A(K, L)

            if (k == l) {
                p = p - (dwork[ncont + l - 1] / dwork[l - 2]) * dwork[ncont2 + l - 1];
                q = q - wr[l - 1] + (dwork[ncont + l - 2] / dwork[l - 2]) * dwork[ncont2 + l - 1];
            }
            p = p / r;
            q = q / r;

            // lp1 already set from when l != ncont
            i32 lp1_val = (l != ncont) ? l + 1 : lp1;
            f64 mp = -p;
            f64 mq = -q;
            SLC_DAXPY(&lp1_val, &mp, b, &int1, &a[(l - 2) * lda], &int1);  // A(1:LP1, L-1)
            SLC_DAXPY(&lp1_val, &mq, b, &int1, &a[(l - 1) * lda], &int1);  // A(1:LP1, L)

            g[l - 2] = p / b1;
            g[l - 1] = q / b1;

            if (l != ncont) {
                ll = 0;
                continue;  // GO TO 20
            }
        }

        break;  // Exit main loop
    }

    // Transform G: G_out = Z * G_in
    // CALL DGEMV('No transpose', NCONT, NCONT, ONE, Z, LDZ, G, 1, ZERO, DWORK, 1)
    SLC_DGEMV("N", &ncont, &ncont, &ONE, z, &ldz, g, &int1, &ZERO, dwork, &int1);
    SLC_DCOPY(&ncont, dwork, &int1, g, &int1);

    // Restore B
    SLC_DSCAL(&ncont, &b1, b, &int1);

    // Annihilate A below first subdiagonal
    if (ncont > 2) {
        i32 nm2 = ncont - 2;
        SLC_DLASET("L", &nm2, &nm2, &ZERO, &ZERO, &a[2], &lda);
    }
}
