/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04PY - Solve discrete-time Sylvester equation with Schur matrices
 *
 * Solves: op(A)*X*op(B) + ISGN*X = scale*C
 *
 * where A and B are in Schur canonical form (block upper triangular with
 * 1x1 and 2x2 diagonal blocks).
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <float.h>

void sb04py(
    const char trana,
    const char tranb,
    const i32 isgn,
    const i32 m,
    const i32 n,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* scale,
    f64* dwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 ibfalse = 0;
    const i32 ibtrue = 1;

    bool notrna, notrnb;
    i32 ierr, j, k, k1, k2, knext, l, l1, l2, lnext;
    i32 mnk1, mnk2, mnl1, mnl2;
    f64 a11, bignum, da11, db, eps, p11, p12, p21, p22;
    f64 scaloc, sgn, smin, smlnum, sumr, xnorm;
    f64 dum[1], vec[4], x[4];

    notrna = (trana == 'N' || trana == 'n');
    notrnb = (tranb == 'N' || tranb == 'n');

    *info = 0;
    if (!notrna && trana != 'T' && trana != 't' && trana != 'C' && trana != 'c') {
        *info = -1;
    } else if (!notrnb && tranb != 'T' && tranb != 't' && tranb != 'C' && tranb != 'c') {
        *info = -2;
    } else if (isgn != 1 && isgn != -1) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < 1 || lda < m) {
        *info = -7;
    } else if (ldb < 1 || ldb < n) {
        *info = -9;
    } else if (ldc < 1 || ldc < m) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    *scale = one;
    if (m == 0 || n == 0) {
        return;
    }

    eps = DBL_EPSILON;
    smlnum = DBL_MIN / eps;
    bignum = one / smlnum;
    SLC_DLABAD(&smlnum, &bignum);
    smlnum = smlnum * (f64)(m * n) / eps;
    bignum = one / smlnum;

    {
        i32 mm = m;
        i32 nn = n;
        f64 anorm = SLC_DLANGE("M", &mm, &mm, a, &lda, dum);
        f64 bnorm = SLC_DLANGE("M", &nn, &nn, b, &ldb, dum);
        smin = eps * anorm;
        if (eps * bnorm > smin) smin = eps * bnorm;
        if (smlnum > smin) smin = smlnum;
    }

    sgn = (f64)isgn;

    #define A(i,j) a[(i) + (j)*lda]
    #define B(i,j) b[(i) + (j)*ldb]
    #define C(i,j) c[(i) + (j)*ldc]
    #define DWORK(i) dwork[(i)]
    #define VEC(i,j) vec[(i) + (j)*2]
    #define X(i,j) x[(i) + (j)*2]

    if (notrna && notrnb) {
        lnext = 0;
        for (l = 0; l < n; l++) {
            if (l < lnext) continue;
            l1 = l;
            if (l == n - 1) {
                l2 = l;
            } else {
                if (B(l+1, l) != zero) {
                    l2 = l + 1;
                } else {
                    l2 = l;
                }
                lnext = l2 + 1;
            }

            knext = m - 1;
            for (k = m - 1; k >= 0; k--) {
                if (k > knext) continue;
                k2 = k;
                if (k == 0) {
                    k1 = k;
                } else {
                    if (A(k, k-1) != zero) {
                        k1 = k - 1;
                    } else {
                        k1 = k;
                    }
                    knext = k1 - 1;
                }

                mnk1 = (k1 + 1 < m) ? k1 + 1 : m;
                mnk2 = (k2 + 1 < m) ? k2 + 1 : m;

                {
                    i32 len = m - k2 - 1;
                    i32 inc1 = lda;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p11 = SLC_DDOT(&len, &A(k1, mnk2), &inc1, &C(mnk2, l1), &inc2);
                    } else {
                        p11 = zero;
                    }
                }

                {
                    i32 len = l1;
                    i32 inc1 = ldc;
                    i32 inc2 = 1;
                    if (len > 0) {
                        DWORK(k1) = SLC_DDOT(&len, &C(k1, 0), &inc1, &B(0, l1), &inc2);
                    } else {
                        DWORK(k1) = zero;
                    }
                }

                if (l1 == l2 && k1 == k2) {
                    i32 len = m - k1;
                    i32 inc1 = lda;
                    i32 inc2 = 1;
                    sumr = SLC_DDOT(&len, &A(k1, k1), &inc1, &DWORK(k1), &inc2);
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1));
                    scaloc = one;

                    a11 = A(k1, k1) * B(l1, l1) + sgn;
                    da11 = fabs(a11);
                    if (da11 <= smin) {
                        a11 = smin;
                        da11 = smin;
                        *info = 1;
                    }
                    db = fabs(VEC(0, 0));
                    if (da11 < one && db > one) {
                        if (db > bignum * da11)
                            scaloc = one / db;
                    }
                    X(0, 0) = (VEC(0, 0) * scaloc) / a11;

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len2 = m - k1;
                        i32 inc = 1;
                        SLC_DSCAL(&len2, &scaloc, &DWORK(k1), &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);

                } else if (l1 == l2 && k1 != k2) {
                    i32 len = m - k2 - 1;
                    i32 inc1 = lda;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p21 = SLC_DDOT(&len, &A(k2, mnk2), &inc1, &C(mnk2, l1), &inc2);
                    } else {
                        p21 = zero;
                    }

                    {
                        i32 len2 = l1;
                        i32 inc1b = ldc;
                        i32 inc2b = 1;
                        if (len2 > 0) {
                            DWORK(k2) = SLC_DDOT(&len2, &C(k2, 0), &inc1b, &B(0, l1), &inc2b);
                        } else {
                            DWORK(k2) = zero;
                        }
                    }

                    {
                        i32 len3 = m - k1;
                        i32 inc1c = lda;
                        i32 inc2c = 1;
                        sumr = SLC_DDOT(&len3, &A(k1, k1), &inc1c, &DWORK(k1), &inc2c);
                    }
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1));

                    {
                        i32 len4 = m - k1;
                        i32 inc1d = lda;
                        i32 inc2d = 1;
                        sumr = SLC_DDOT(&len4, &A(k2, k1), &inc1d, &DWORK(k1), &inc2d);
                    }
                    VEC(1, 0) = C(k2, l1) - (sumr + p21 * B(l1, l1));

                    {
                        i32 n1 = 2, n2 = 1;
                        f64 d1 = one, d2 = one;
                        f64 neg_sgn = -sgn;
                        SLC_DLALN2(&ibfalse, &n1, &n2, &smin, &B(l1, l1),
                                   &A(k1, k1), &lda, &d1, &d2, vec, &n1,
                                   &neg_sgn, &zero, x, &n1, &scaloc, &xnorm, &ierr);
                        if (ierr != 0) *info = 1;
                    }

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len5 = m - k1;
                        i32 inc = 1;
                        SLC_DSCAL(&len5, &scaloc, &DWORK(k1), &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);
                    C(k2, l1) = X(1, 0);

                } else if (l1 != l2 && k1 == k2) {
                    i32 len = m - k1 - 1;
                    i32 inc1 = lda;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p12 = SLC_DDOT(&len, &A(k1, mnk1), &inc1, &C(mnk1, l2), &inc2);
                    } else {
                        p12 = zero;
                    }

                    {
                        i32 len2 = m - k1;
                        i32 inc1b = lda;
                        i32 inc2b = 1;
                        sumr = SLC_DDOT(&len2, &A(k1, k1), &inc1b, &DWORK(k1), &inc2b);
                    }
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1) + p12 * B(l2, l1));

                    {
                        i32 len3 = l1;
                        i32 inc1c = ldc;
                        i32 inc2c = 1;
                        if (len3 > 0) {
                            DWORK(k1 + m) = SLC_DDOT(&len3, &C(k1, 0), &inc1c, &B(0, l2), &inc2c);
                        } else {
                            DWORK(k1 + m) = zero;
                        }
                    }

                    {
                        i32 len4 = m - k1;
                        i32 inc1d = lda;
                        i32 inc2d = 1;
                        sumr = SLC_DDOT(&len4, &A(k1, k1), &inc1d, &DWORK(k1 + m), &inc2d);
                    }
                    VEC(1, 0) = C(k1, l2) - (sumr + p11 * B(l1, l2) + p12 * B(l2, l2));

                    {
                        i32 n1 = 2, n2 = 1;
                        f64 d1 = one, d2 = one;
                        f64 neg_sgn = -sgn;
                        SLC_DLALN2(&ibtrue, &n1, &n2, &smin, &A(k1, k1),
                                   &B(l1, l1), &ldb, &d1, &d2, vec, &n1,
                                   &neg_sgn, &zero, x, &n1, &scaloc, &xnorm, &ierr);
                        if (ierr != 0) *info = 1;
                    }

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len5 = m - k1;
                        i32 inc = 1;
                        SLC_DSCAL(&len5, &scaloc, &DWORK(k1), &inc);
                        SLC_DSCAL(&len5, &scaloc, &DWORK(k1 + m), &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);
                    C(k1, l2) = X(1, 0);

                } else {
                    i32 len = m - k2 - 1;
                    i32 inc1 = lda;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p21 = SLC_DDOT(&len, &A(k2, mnk2), &inc1, &C(mnk2, l1), &inc2);
                        p12 = SLC_DDOT(&len, &A(k1, mnk2), &inc1, &C(mnk2, l2), &inc2);
                        p22 = SLC_DDOT(&len, &A(k2, mnk2), &inc1, &C(mnk2, l2), &inc2);
                    } else {
                        p21 = zero;
                        p12 = zero;
                        p22 = zero;
                    }

                    {
                        i32 len2 = l1;
                        i32 inc1b = ldc;
                        i32 inc2b = 1;
                        if (len2 > 0) {
                            DWORK(k2) = SLC_DDOT(&len2, &C(k2, 0), &inc1b, &B(0, l1), &inc2b);
                        } else {
                            DWORK(k2) = zero;
                        }
                    }

                    {
                        i32 len3 = m - k1;
                        i32 inc1c = lda;
                        i32 inc2c = 1;
                        sumr = SLC_DDOT(&len3, &A(k1, k1), &inc1c, &DWORK(k1), &inc2c);
                    }
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1) + p12 * B(l2, l1));

                    {
                        i32 len4 = l1;
                        i32 inc1d = ldc;
                        i32 inc2d = 1;
                        if (len4 > 0) {
                            DWORK(k1 + m) = SLC_DDOT(&len4, &C(k1, 0), &inc1d, &B(0, l2), &inc2d);
                            DWORK(k2 + m) = SLC_DDOT(&len4, &C(k2, 0), &inc1d, &B(0, l2), &inc2d);
                        } else {
                            DWORK(k1 + m) = zero;
                            DWORK(k2 + m) = zero;
                        }
                    }

                    {
                        i32 len5 = m - k1;
                        i32 inc1e = lda;
                        i32 inc2e = 1;
                        sumr = SLC_DDOT(&len5, &A(k1, k1), &inc1e, &DWORK(k1 + m), &inc2e);
                    }
                    VEC(0, 1) = C(k1, l2) - (sumr + p11 * B(l1, l2) + p12 * B(l2, l2));

                    {
                        i32 len6 = m - k1;
                        i32 inc1f = lda;
                        i32 inc2f = 1;
                        sumr = SLC_DDOT(&len6, &A(k2, k1), &inc1f, &DWORK(k1), &inc2f);
                    }
                    VEC(1, 0) = C(k2, l1) - (sumr + p21 * B(l1, l1) + p22 * B(l2, l1));

                    {
                        i32 len7 = m - k1;
                        i32 inc1g = lda;
                        i32 inc2g = 1;
                        sumr = SLC_DDOT(&len7, &A(k2, k1), &inc1g, &DWORK(k1 + m), &inc2g);
                    }
                    VEC(1, 1) = C(k2, l2) - (sumr + p21 * B(l1, l2) + p22 * B(l2, l2));

                    {
                        i32 n1 = 2, n2 = 2;
                        sb04px(false, false, isgn, n1, n2,
                               &A(k1, k1), lda, &B(l1, l1), ldb,
                               vec, n1, &scaloc, x, n1, &xnorm, &ierr);
                        if (ierr != 0) *info = 1;
                    }

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len8 = m - k1;
                        i32 inc = 1;
                        SLC_DSCAL(&len8, &scaloc, &DWORK(k1), &inc);
                        SLC_DSCAL(&len8, &scaloc, &DWORK(k1 + m), &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);
                    C(k1, l2) = X(0, 1);
                    C(k2, l1) = X(1, 0);
                    C(k2, l2) = X(1, 1);
                }
            }
        }

    } else if (!notrna && notrnb) {
        lnext = 0;
        for (l = 0; l < n; l++) {
            if (l < lnext) continue;
            l1 = l;
            if (l == n - 1) {
                l2 = l;
            } else {
                if (B(l+1, l) != zero) {
                    l2 = l + 1;
                } else {
                    l2 = l;
                }
                lnext = l2 + 1;
            }

            knext = 0;
            for (k = 0; k < m; k++) {
                if (k < knext) continue;
                k1 = k;
                if (k == m - 1) {
                    k2 = k;
                } else {
                    if (A(k+1, k) != zero) {
                        k2 = k + 1;
                    } else {
                        k2 = k;
                    }
                    knext = k2 + 1;
                }

                {
                    i32 len = k1;
                    i32 inc1 = 1;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p11 = SLC_DDOT(&len, &A(0, k1), &inc1, &C(0, l1), &inc2);
                    } else {
                        p11 = zero;
                    }
                }

                {
                    i32 len = l1;
                    i32 inc1 = ldc;
                    i32 inc2 = 1;
                    if (len > 0) {
                        DWORK(k1) = SLC_DDOT(&len, &C(k1, 0), &inc1, &B(0, l1), &inc2);
                    } else {
                        DWORK(k1) = zero;
                    }
                }

                if (l1 == l2 && k1 == k2) {
                    i32 len = k1 + 1;
                    i32 inc1 = 1;
                    i32 inc2 = 1;
                    sumr = SLC_DDOT(&len, &A(0, k1), &inc1, dwork, &inc2);
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1));
                    scaloc = one;

                    a11 = A(k1, k1) * B(l1, l1) + sgn;
                    da11 = fabs(a11);
                    if (da11 <= smin) {
                        a11 = smin;
                        da11 = smin;
                        *info = 1;
                    }
                    db = fabs(VEC(0, 0));
                    if (da11 < one && db > one) {
                        if (db > bignum * da11)
                            scaloc = one / db;
                    }
                    X(0, 0) = (VEC(0, 0) * scaloc) / a11;

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len2 = k1 + 1;
                        i32 inc = 1;
                        SLC_DSCAL(&len2, &scaloc, dwork, &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);

                } else if (l1 == l2 && k1 != k2) {
                    i32 len = k1;
                    i32 inc1 = 1;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p21 = SLC_DDOT(&len, &A(0, k2), &inc1, &C(0, l1), &inc2);
                    } else {
                        p21 = zero;
                    }

                    {
                        i32 len2 = l1;
                        i32 inc1b = ldc;
                        i32 inc2b = 1;
                        if (len2 > 0) {
                            DWORK(k2) = SLC_DDOT(&len2, &C(k2, 0), &inc1b, &B(0, l1), &inc2b);
                        } else {
                            DWORK(k2) = zero;
                        }
                    }

                    {
                        i32 len3 = k2 + 1;
                        i32 inc1c = 1;
                        i32 inc2c = 1;
                        sumr = SLC_DDOT(&len3, &A(0, k1), &inc1c, dwork, &inc2c);
                    }
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1));

                    {
                        i32 len4 = k2 + 1;
                        i32 inc1d = 1;
                        i32 inc2d = 1;
                        sumr = SLC_DDOT(&len4, &A(0, k2), &inc1d, dwork, &inc2d);
                    }
                    VEC(1, 0) = C(k2, l1) - (sumr + p21 * B(l1, l1));

                    {
                        i32 n1 = 2, n2 = 1;
                        f64 d1 = one, d2 = one;
                        f64 neg_sgn = -sgn;
                        SLC_DLALN2(&ibtrue, &n1, &n2, &smin, &B(l1, l1),
                                   &A(k1, k1), &lda, &d1, &d2, vec, &n1,
                                   &neg_sgn, &zero, x, &n1, &scaloc, &xnorm, &ierr);
                        if (ierr != 0) *info = 1;
                    }

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len5 = k2 + 1;
                        i32 inc = 1;
                        SLC_DSCAL(&len5, &scaloc, dwork, &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);
                    C(k2, l1) = X(1, 0);

                } else if (l1 != l2 && k1 == k2) {
                    i32 len = k1;
                    i32 inc1 = 1;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p12 = SLC_DDOT(&len, &A(0, k1), &inc1, &C(0, l2), &inc2);
                    } else {
                        p12 = zero;
                    }

                    {
                        i32 len2 = k1 + 1;
                        i32 inc1b = 1;
                        i32 inc2b = 1;
                        sumr = SLC_DDOT(&len2, &A(0, k1), &inc1b, dwork, &inc2b);
                    }
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1) + p12 * B(l2, l1));

                    {
                        i32 len3 = l1;
                        i32 inc1c = ldc;
                        i32 inc2c = 1;
                        if (len3 > 0) {
                            DWORK(k1 + m) = SLC_DDOT(&len3, &C(k1, 0), &inc1c, &B(0, l2), &inc2c);
                        } else {
                            DWORK(k1 + m) = zero;
                        }
                    }

                    {
                        i32 len4 = k1 + 1;
                        i32 inc1d = 1;
                        i32 inc2d = 1;
                        sumr = SLC_DDOT(&len4, &A(0, k1), &inc1d, &DWORK(m), &inc2d);
                    }
                    VEC(1, 0) = C(k1, l2) - (sumr + p11 * B(l1, l2) + p12 * B(l2, l2));

                    {
                        i32 n1 = 2, n2 = 1;
                        f64 d1 = one, d2 = one;
                        f64 neg_sgn = -sgn;
                        SLC_DLALN2(&ibtrue, &n1, &n2, &smin, &A(k1, k1),
                                   &B(l1, l1), &ldb, &d1, &d2, vec, &n1,
                                   &neg_sgn, &zero, x, &n1, &scaloc, &xnorm, &ierr);
                        if (ierr != 0) *info = 1;
                    }

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len5 = k1 + 1;
                        i32 inc = 1;
                        SLC_DSCAL(&len5, &scaloc, dwork, &inc);
                        SLC_DSCAL(&len5, &scaloc, &DWORK(m), &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);
                    C(k1, l2) = X(1, 0);

                } else {
                    i32 len = k1;
                    i32 inc1 = 1;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p21 = SLC_DDOT(&len, &A(0, k2), &inc1, &C(0, l1), &inc2);
                        p12 = SLC_DDOT(&len, &A(0, k1), &inc1, &C(0, l2), &inc2);
                        p22 = SLC_DDOT(&len, &A(0, k2), &inc1, &C(0, l2), &inc2);
                    } else {
                        p21 = zero;
                        p12 = zero;
                        p22 = zero;
                    }

                    {
                        i32 len2 = l1;
                        i32 inc1b = ldc;
                        i32 inc2b = 1;
                        if (len2 > 0) {
                            DWORK(k2) = SLC_DDOT(&len2, &C(k2, 0), &inc1b, &B(0, l1), &inc2b);
                        } else {
                            DWORK(k2) = zero;
                        }
                    }

                    {
                        i32 len3 = k2 + 1;
                        i32 inc1c = 1;
                        i32 inc2c = 1;
                        sumr = SLC_DDOT(&len3, &A(0, k1), &inc1c, dwork, &inc2c);
                    }
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1) + p12 * B(l2, l1));

                    {
                        i32 len4 = k2 + 1;
                        i32 inc1d = 1;
                        i32 inc2d = 1;
                        sumr = SLC_DDOT(&len4, &A(0, k2), &inc1d, dwork, &inc2d);
                    }
                    VEC(1, 0) = C(k2, l1) - (sumr + p21 * B(l1, l1) + p22 * B(l2, l1));

                    {
                        i32 len5 = l1;
                        i32 inc1e = ldc;
                        i32 inc2e = 1;
                        if (len5 > 0) {
                            DWORK(k1 + m) = SLC_DDOT(&len5, &C(k1, 0), &inc1e, &B(0, l2), &inc2e);
                            DWORK(k2 + m) = SLC_DDOT(&len5, &C(k2, 0), &inc1e, &B(0, l2), &inc2e);
                        } else {
                            DWORK(k1 + m) = zero;
                            DWORK(k2 + m) = zero;
                        }
                    }

                    {
                        i32 len6 = k2 + 1;
                        i32 inc1f = 1;
                        i32 inc2f = 1;
                        sumr = SLC_DDOT(&len6, &A(0, k1), &inc1f, &DWORK(m), &inc2f);
                    }
                    VEC(0, 1) = C(k1, l2) - (sumr + p11 * B(l1, l2) + p12 * B(l2, l2));

                    {
                        i32 len7 = k2 + 1;
                        i32 inc1g = 1;
                        i32 inc2g = 1;
                        sumr = SLC_DDOT(&len7, &A(0, k2), &inc1g, &DWORK(m), &inc2g);
                    }
                    VEC(1, 1) = C(k2, l2) - (sumr + p21 * B(l1, l2) + p22 * B(l2, l2));

                    {
                        i32 n1 = 2, n2 = 2;
                        sb04px(true, false, isgn, n1, n2,
                               &A(k1, k1), lda, &B(l1, l1), ldb,
                               vec, n1, &scaloc, x, n1, &xnorm, &ierr);
                        if (ierr != 0) *info = 1;
                    }

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len8 = k2 + 1;
                        i32 inc = 1;
                        SLC_DSCAL(&len8, &scaloc, dwork, &inc);
                        SLC_DSCAL(&len8, &scaloc, &DWORK(m), &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);
                    C(k1, l2) = X(0, 1);
                    C(k2, l1) = X(1, 0);
                    C(k2, l2) = X(1, 1);
                }
            }
        }

    } else if (!notrna && !notrnb) {
        lnext = n - 1;
        for (l = n - 1; l >= 0; l--) {
            if (l > lnext) continue;
            l2 = l;
            if (l == 0) {
                l1 = l;
            } else {
                if (B(l, l-1) != zero) {
                    l1 = l - 1;
                } else {
                    l1 = l;
                }
                lnext = l1 - 1;
            }

            knext = 0;
            for (k = 0; k < m; k++) {
                if (k < knext) continue;
                k1 = k;
                if (k == m - 1) {
                    k2 = k;
                } else {
                    if (A(k+1, k) != zero) {
                        k2 = k + 1;
                    } else {
                        k2 = k;
                    }
                    knext = k2 + 1;
                }

                mnl1 = (l1 + 1 < n) ? l1 + 1 : n;
                mnl2 = (l2 + 1 < n) ? l2 + 1 : n;

                {
                    i32 len = k1;
                    i32 inc1 = 1;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p11 = SLC_DDOT(&len, &A(0, k1), &inc1, &C(0, l1), &inc2);
                    } else {
                        p11 = zero;
                    }
                }

                {
                    i32 len = n - l2 - 1;
                    i32 inc1 = ldc;
                    i32 inc2 = ldb;
                    if (len > 0) {
                        DWORK(k1) = SLC_DDOT(&len, &C(k1, mnl2), &inc1, &B(l1, mnl2), &inc2);
                    } else {
                        DWORK(k1) = zero;
                    }
                }

                if (l1 == l2 && k1 == k2) {
                    i32 len = k1 + 1;
                    i32 inc1 = 1;
                    i32 inc2 = 1;
                    sumr = SLC_DDOT(&len, &A(0, k1), &inc1, dwork, &inc2);
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1));
                    scaloc = one;

                    a11 = A(k1, k1) * B(l1, l1) + sgn;
                    da11 = fabs(a11);
                    if (da11 <= smin) {
                        a11 = smin;
                        da11 = smin;
                        *info = 1;
                    }
                    db = fabs(VEC(0, 0));
                    if (da11 < one && db > one) {
                        if (db > bignum * da11)
                            scaloc = one / db;
                    }
                    X(0, 0) = (VEC(0, 0) * scaloc) / a11;

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len2 = k1 + 1;
                        i32 inc = 1;
                        SLC_DSCAL(&len2, &scaloc, dwork, &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);

                } else if (l1 == l2 && k1 != k2) {
                    i32 len = k1;
                    i32 inc1 = 1;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p21 = SLC_DDOT(&len, &A(0, k2), &inc1, &C(0, l1), &inc2);
                    } else {
                        p21 = zero;
                    }

                    {
                        i32 len2 = n - l1 - 1;
                        i32 inc1b = ldc;
                        i32 inc2b = ldb;
                        if (len2 > 0) {
                            DWORK(k2) = SLC_DDOT(&len2, &C(k2, mnl1), &inc1b, &B(l1, mnl1), &inc2b);
                        } else {
                            DWORK(k2) = zero;
                        }
                    }

                    {
                        i32 len3 = k2 + 1;
                        i32 inc1c = 1;
                        i32 inc2c = 1;
                        sumr = SLC_DDOT(&len3, &A(0, k1), &inc1c, dwork, &inc2c);
                    }
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1));

                    {
                        i32 len4 = k2 + 1;
                        i32 inc1d = 1;
                        i32 inc2d = 1;
                        sumr = SLC_DDOT(&len4, &A(0, k2), &inc1d, dwork, &inc2d);
                    }
                    VEC(1, 0) = C(k2, l1) - (sumr + p21 * B(l1, l1));

                    {
                        i32 n1 = 2, n2 = 1;
                        f64 d1 = one, d2 = one;
                        f64 neg_sgn = -sgn;
                        SLC_DLALN2(&ibtrue, &n1, &n2, &smin, &B(l1, l1),
                                   &A(k1, k1), &lda, &d1, &d2, vec, &n1,
                                   &neg_sgn, &zero, x, &n1, &scaloc, &xnorm, &ierr);
                        if (ierr != 0) *info = 1;
                    }

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len5 = k2 + 1;
                        i32 inc = 1;
                        SLC_DSCAL(&len5, &scaloc, dwork, &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);
                    C(k2, l1) = X(1, 0);

                } else if (l1 != l2 && k1 == k2) {
                    i32 len = k1;
                    i32 inc1 = 1;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p12 = SLC_DDOT(&len, &A(0, k1), &inc1, &C(0, l2), &inc2);
                    } else {
                        p12 = zero;
                    }

                    {
                        i32 len2 = k1 + 1;
                        i32 inc1b = 1;
                        i32 inc2b = 1;
                        sumr = SLC_DDOT(&len2, &A(0, k1), &inc1b, dwork, &inc2b);
                    }
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1) + p12 * B(l1, l2));

                    {
                        i32 len3 = n - l2 - 1;
                        i32 inc1c = ldc;
                        i32 inc2c = ldb;
                        if (len3 > 0) {
                            DWORK(k1 + m) = SLC_DDOT(&len3, &C(k1, mnl2), &inc1c, &B(l2, mnl2), &inc2c);
                        } else {
                            DWORK(k1 + m) = zero;
                        }
                    }

                    {
                        i32 len4 = k1 + 1;
                        i32 inc1d = 1;
                        i32 inc2d = 1;
                        sumr = SLC_DDOT(&len4, &A(0, k1), &inc1d, &DWORK(m), &inc2d);
                    }
                    VEC(1, 0) = C(k1, l2) - (sumr + p11 * B(l2, l1) + p12 * B(l2, l2));

                    {
                        i32 n1 = 2, n2 = 1;
                        f64 d1 = one, d2 = one;
                        f64 neg_sgn = -sgn;
                        SLC_DLALN2(&ibfalse, &n1, &n2, &smin, &A(k1, k1),
                                   &B(l1, l1), &ldb, &d1, &d2, vec, &n1,
                                   &neg_sgn, &zero, x, &n1, &scaloc, &xnorm, &ierr);
                        if (ierr != 0) *info = 1;
                    }

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len5 = k1 + 1;
                        i32 inc = 1;
                        SLC_DSCAL(&len5, &scaloc, dwork, &inc);
                        SLC_DSCAL(&len5, &scaloc, &DWORK(m), &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);
                    C(k1, l2) = X(1, 0);

                } else {
                    i32 len = k1;
                    i32 inc1 = 1;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p21 = SLC_DDOT(&len, &A(0, k2), &inc1, &C(0, l1), &inc2);
                        p12 = SLC_DDOT(&len, &A(0, k1), &inc1, &C(0, l2), &inc2);
                        p22 = SLC_DDOT(&len, &A(0, k2), &inc1, &C(0, l2), &inc2);
                    } else {
                        p21 = zero;
                        p12 = zero;
                        p22 = zero;
                    }

                    {
                        i32 len2 = n - l2 - 1;
                        i32 inc1b = ldc;
                        i32 inc2b = ldb;
                        if (len2 > 0) {
                            DWORK(k2) = SLC_DDOT(&len2, &C(k2, mnl2), &inc1b, &B(l1, mnl2), &inc2b);
                        } else {
                            DWORK(k2) = zero;
                        }
                    }

                    {
                        i32 len3 = k2 + 1;
                        i32 inc1c = 1;
                        i32 inc2c = 1;
                        sumr = SLC_DDOT(&len3, &A(0, k1), &inc1c, dwork, &inc2c);
                    }
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1) + p12 * B(l1, l2));

                    {
                        i32 len4 = k2 + 1;
                        i32 inc1d = 1;
                        i32 inc2d = 1;
                        sumr = SLC_DDOT(&len4, &A(0, k2), &inc1d, dwork, &inc2d);
                    }
                    VEC(1, 0) = C(k2, l1) - (sumr + p21 * B(l1, l1) + p22 * B(l1, l2));

                    {
                        i32 len5 = n - l2 - 1;
                        i32 inc1e = ldc;
                        i32 inc2e = ldb;
                        if (len5 > 0) {
                            DWORK(k1 + m) = SLC_DDOT(&len5, &C(k1, mnl2), &inc1e, &B(l2, mnl2), &inc2e);
                            DWORK(k2 + m) = SLC_DDOT(&len5, &C(k2, mnl2), &inc1e, &B(l2, mnl2), &inc2e);
                        } else {
                            DWORK(k1 + m) = zero;
                            DWORK(k2 + m) = zero;
                        }
                    }

                    {
                        i32 len6 = k2 + 1;
                        i32 inc1f = 1;
                        i32 inc2f = 1;
                        sumr = SLC_DDOT(&len6, &A(0, k1), &inc1f, &DWORK(m), &inc2f);
                    }
                    VEC(0, 1) = C(k1, l2) - (sumr + p11 * B(l2, l1) + p12 * B(l2, l2));

                    {
                        i32 len7 = k2 + 1;
                        i32 inc1g = 1;
                        i32 inc2g = 1;
                        sumr = SLC_DDOT(&len7, &A(0, k2), &inc1g, &DWORK(m), &inc2g);
                    }
                    VEC(1, 1) = C(k2, l2) - (sumr + p21 * B(l2, l1) + p22 * B(l2, l2));

                    {
                        i32 n1 = 2, n2 = 2;
                        sb04px(true, true, isgn, n1, n2,
                               &A(k1, k1), lda, &B(l1, l1), ldb,
                               vec, n1, &scaloc, x, n1, &xnorm, &ierr);
                        if (ierr != 0) *info = 1;
                    }

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len8 = k2 + 1;
                        i32 inc = 1;
                        SLC_DSCAL(&len8, &scaloc, dwork, &inc);
                        SLC_DSCAL(&len8, &scaloc, &DWORK(m), &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);
                    C(k1, l2) = X(0, 1);
                    C(k2, l1) = X(1, 0);
                    C(k2, l2) = X(1, 1);
                }
            }
        }

    } else {
        lnext = n - 1;
        for (l = n - 1; l >= 0; l--) {
            if (l > lnext) continue;
            l2 = l;
            if (l == 0) {
                l1 = l;
            } else {
                if (B(l, l-1) != zero) {
                    l1 = l - 1;
                } else {
                    l1 = l;
                }
                lnext = l1 - 1;
            }

            knext = m - 1;
            for (k = m - 1; k >= 0; k--) {
                if (k > knext) continue;
                k2 = k;
                if (k == 0) {
                    k1 = k;
                } else {
                    if (A(k, k-1) != zero) {
                        k1 = k - 1;
                    } else {
                        k1 = k;
                    }
                    knext = k1 - 1;
                }

                mnk1 = (k1 + 1 < m) ? k1 + 1 : m;
                mnk2 = (k2 + 1 < m) ? k2 + 1 : m;
                mnl1 = (l1 + 1 < n) ? l1 + 1 : n;
                mnl2 = (l2 + 1 < n) ? l2 + 1 : n;

                {
                    i32 len = m - k2 - 1;
                    i32 inc1 = lda;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p11 = SLC_DDOT(&len, &A(k1, mnk2), &inc1, &C(mnk2, l1), &inc2);
                    } else {
                        p11 = zero;
                    }
                }

                {
                    i32 len = n - l2 - 1;
                    i32 inc1 = ldc;
                    i32 inc2 = ldb;
                    if (len > 0) {
                        DWORK(k1) = SLC_DDOT(&len, &C(k1, mnl2), &inc1, &B(l1, mnl2), &inc2);
                    } else {
                        DWORK(k1) = zero;
                    }
                }

                if (l1 == l2 && k1 == k2) {
                    i32 len = m - k1;
                    i32 inc1 = lda;
                    i32 inc2 = 1;
                    sumr = SLC_DDOT(&len, &A(k1, k1), &inc1, &DWORK(k1), &inc2);
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1));
                    scaloc = one;

                    a11 = A(k1, k1) * B(l1, l1) + sgn;
                    da11 = fabs(a11);
                    if (da11 <= smin) {
                        a11 = smin;
                        da11 = smin;
                        *info = 1;
                    }
                    db = fabs(VEC(0, 0));
                    if (da11 < one && db > one) {
                        if (db > bignum * da11)
                            scaloc = one / db;
                    }
                    X(0, 0) = (VEC(0, 0) * scaloc) / a11;

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len2 = m - k1;
                        i32 inc = 1;
                        SLC_DSCAL(&len2, &scaloc, &DWORK(k1), &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);

                } else if (l1 == l2 && k1 != k2) {
                    i32 len = m - k2 - 1;
                    i32 inc1 = lda;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p21 = SLC_DDOT(&len, &A(k2, mnk2), &inc1, &C(mnk2, l1), &inc2);
                    } else {
                        p21 = zero;
                    }

                    {
                        i32 len2 = n - l1 - 1;
                        i32 inc1b = ldc;
                        i32 inc2b = ldb;
                        if (len2 > 0) {
                            DWORK(k2) = SLC_DDOT(&len2, &C(k2, mnl1), &inc1b, &B(l1, mnl1), &inc2b);
                        } else {
                            DWORK(k2) = zero;
                        }
                    }

                    {
                        i32 len3 = m - k1;
                        i32 inc1c = lda;
                        i32 inc2c = 1;
                        sumr = SLC_DDOT(&len3, &A(k1, k1), &inc1c, &DWORK(k1), &inc2c);
                    }
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1));

                    {
                        i32 len4 = m - k1;
                        i32 inc1d = lda;
                        i32 inc2d = 1;
                        sumr = SLC_DDOT(&len4, &A(k2, k1), &inc1d, &DWORK(k1), &inc2d);
                    }
                    VEC(1, 0) = C(k2, l1) - (sumr + p21 * B(l1, l1));

                    {
                        i32 n1 = 2, n2 = 1;
                        f64 d1 = one, d2 = one;
                        f64 neg_sgn = -sgn;
                        SLC_DLALN2(&ibfalse, &n1, &n2, &smin, &B(l1, l1),
                                   &A(k1, k1), &lda, &d1, &d2, vec, &n1,
                                   &neg_sgn, &zero, x, &n1, &scaloc, &xnorm, &ierr);
                        if (ierr != 0) *info = 1;
                    }

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len5 = m - k1;
                        i32 inc = 1;
                        SLC_DSCAL(&len5, &scaloc, &DWORK(k1), &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);
                    C(k2, l1) = X(1, 0);

                } else if (l1 != l2 && k1 == k2) {
                    i32 len = m - k1 - 1;
                    i32 inc1 = lda;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p12 = SLC_DDOT(&len, &A(k1, mnk1), &inc1, &C(mnk1, l2), &inc2);
                    } else {
                        p12 = zero;
                    }

                    {
                        i32 len2 = m - k1;
                        i32 inc1b = lda;
                        i32 inc2b = 1;
                        sumr = SLC_DDOT(&len2, &A(k1, k1), &inc1b, &DWORK(k1), &inc2b);
                    }
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1) + p12 * B(l1, l2));

                    {
                        i32 len3 = n - l2 - 1;
                        i32 inc1c = ldc;
                        i32 inc2c = ldb;
                        if (len3 > 0) {
                            DWORK(k1 + m) = SLC_DDOT(&len3, &C(k1, mnl2), &inc1c, &B(l2, mnl2), &inc2c);
                        } else {
                            DWORK(k1 + m) = zero;
                        }
                    }

                    {
                        i32 len4 = m - k1;
                        i32 inc1d = lda;
                        i32 inc2d = 1;
                        sumr = SLC_DDOT(&len4, &A(k1, k1), &inc1d, &DWORK(k1 + m), &inc2d);
                    }
                    VEC(1, 0) = C(k1, l2) - (sumr + p11 * B(l2, l1) + p12 * B(l2, l2));

                    {
                        i32 n1 = 2, n2 = 1;
                        f64 d1 = one, d2 = one;
                        f64 neg_sgn = -sgn;
                        SLC_DLALN2(&ibfalse, &n1, &n2, &smin, &A(k1, k1),
                                   &B(l1, l1), &ldb, &d1, &d2, vec, &n1,
                                   &neg_sgn, &zero, x, &n1, &scaloc, &xnorm, &ierr);
                        if (ierr != 0) *info = 1;
                    }

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len5 = m - k1;
                        i32 inc = 1;
                        SLC_DSCAL(&len5, &scaloc, &DWORK(k1), &inc);
                        SLC_DSCAL(&len5, &scaloc, &DWORK(k1 + m), &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);
                    C(k1, l2) = X(1, 0);

                } else {
                    i32 len = m - k2 - 1;
                    i32 inc1 = lda;
                    i32 inc2 = 1;
                    if (len > 0) {
                        p21 = SLC_DDOT(&len, &A(k2, mnk2), &inc1, &C(mnk2, l1), &inc2);
                        p12 = SLC_DDOT(&len, &A(k1, mnk2), &inc1, &C(mnk2, l2), &inc2);
                        p22 = SLC_DDOT(&len, &A(k2, mnk2), &inc1, &C(mnk2, l2), &inc2);
                    } else {
                        p21 = zero;
                        p12 = zero;
                        p22 = zero;
                    }

                    {
                        i32 len2 = n - l2 - 1;
                        i32 inc1b = ldc;
                        i32 inc2b = ldb;
                        if (len2 > 0) {
                            DWORK(k2) = SLC_DDOT(&len2, &C(k2, mnl2), &inc1b, &B(l1, mnl2), &inc2b);
                        } else {
                            DWORK(k2) = zero;
                        }
                    }

                    {
                        i32 len3 = m - k1;
                        i32 inc1c = lda;
                        i32 inc2c = 1;
                        sumr = SLC_DDOT(&len3, &A(k1, k1), &inc1c, &DWORK(k1), &inc2c);
                    }
                    VEC(0, 0) = C(k1, l1) - (sumr + p11 * B(l1, l1) + p12 * B(l1, l2));

                    {
                        i32 len4 = m - k1;
                        i32 inc1d = lda;
                        i32 inc2d = 1;
                        sumr = SLC_DDOT(&len4, &A(k2, k1), &inc1d, &DWORK(k1), &inc2d);
                    }
                    VEC(1, 0) = C(k2, l1) - (sumr + p21 * B(l1, l1) + p22 * B(l1, l2));

                    {
                        i32 len5 = n - l2 - 1;
                        i32 inc1e = ldc;
                        i32 inc2e = ldb;
                        if (len5 > 0) {
                            DWORK(k1 + m) = SLC_DDOT(&len5, &C(k1, mnl2), &inc1e, &B(l2, mnl2), &inc2e);
                            DWORK(k2 + m) = SLC_DDOT(&len5, &C(k2, mnl2), &inc1e, &B(l2, mnl2), &inc2e);
                        } else {
                            DWORK(k1 + m) = zero;
                            DWORK(k2 + m) = zero;
                        }
                    }

                    {
                        i32 len6 = m - k1;
                        i32 inc1f = lda;
                        i32 inc2f = 1;
                        sumr = SLC_DDOT(&len6, &A(k1, k1), &inc1f, &DWORK(k1 + m), &inc2f);
                    }
                    VEC(0, 1) = C(k1, l2) - (sumr + p11 * B(l2, l1) + p12 * B(l2, l2));

                    {
                        i32 len7 = m - k1;
                        i32 inc1g = lda;
                        i32 inc2g = 1;
                        sumr = SLC_DDOT(&len7, &A(k2, k1), &inc1g, &DWORK(k1 + m), &inc2g);
                    }
                    VEC(1, 1) = C(k2, l2) - (sumr + p21 * B(l2, l1) + p22 * B(l2, l2));

                    {
                        i32 n1 = 2, n2 = 2;
                        sb04px(false, true, isgn, n1, n2,
                               &A(k1, k1), lda, &B(l1, l1), ldb,
                               vec, n1, &scaloc, x, n1, &xnorm, &ierr);
                        if (ierr != 0) *info = 1;
                    }

                    if (scaloc != one) {
                        for (j = 0; j < n; j++) {
                            i32 mm = m;
                            i32 inc = 1;
                            SLC_DSCAL(&mm, &scaloc, &C(0, j), &inc);
                        }
                        i32 len8 = m - k1;
                        i32 inc = 1;
                        SLC_DSCAL(&len8, &scaloc, &DWORK(k1), &inc);
                        SLC_DSCAL(&len8, &scaloc, &DWORK(k1 + m), &inc);
                        *scale *= scaloc;
                    }
                    C(k1, l1) = X(0, 0);
                    C(k1, l2) = X(0, 1);
                    C(k2, l1) = X(1, 0);
                    C(k2, l2) = X(1, 1);
                }
            }
        }
    }

    #undef A
    #undef B
    #undef C
    #undef DWORK
    #undef VEC
    #undef X
}
