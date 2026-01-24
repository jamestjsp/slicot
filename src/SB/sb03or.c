/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03OR - Solve real quasi-triangular Sylvester equation
 *
 * Solves for N-by-M matrix X (M = 1 or 2) in:
 *   op(S)'*X + X*op(A) = scale*C   (DISCR = false, continuous)
 *   op(S)'*X*op(A) - X = scale*C   (DISCR = true, discrete)
 *
 * where op(K) = K or K', S is N-by-N block upper triangular,
 * A is M-by-M. Solution X overwrites C.
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb03or(
    const bool discr,
    const bool ltrans,
    const i32 n,
    const i32 m,
    const f64* s,
    const i32 lds,
    const f64* a,
    const i32 lda,
    f64* c,
    const i32 ldc,
    f64* scale,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    f64 at[4];
    f64 vec[4];
    f64 x[4];

    i32 dl, l, l1, l2, l2p1, lnext;
    i32 isgn, infom;
    const i32 one_inc = 1;
    f64 g11, g12, g21, g22, scaloc, xnorm;
    bool tbyt;

    *info = 0;

    if (n < 0) {
        *info = -3;
        return;
    }
    if (m != 1 && m != 2) {
        *info = -4;
        return;
    }
    if (lds < (n > 1 ? n : 1)) {
        *info = -6;
        return;
    }
    if (lda < m) {
        *info = -8;
        return;
    }
    if (ldc < (n > 1 ? n : 1)) {
        *info = -10;
        return;
    }

    *scale = one;

    if (n == 0) {
        return;
    }

    isgn = 1;
    tbyt = (m == 2);
    infom = 0;

    at[0] = a[0];
    if (tbyt) {
        at[1] = a[0 + 1*lda];
        at[2] = a[1];
        at[3] = a[1 + 1*lda];
    }

    if (ltrans) {
        lnext = n;

        for (l = n; l >= 1; l--) {
            if (l > lnext) {
                continue;
            }
            l1 = l;
            l2 = l;
            if (l > 1) {
                if (s[(l-1) + (l-2)*lds] != zero) {
                    l1 = l1 - 1;
                }
                lnext = l1 - 1;
            }
            dl = l2 - l1 + 1;
            l2p1 = (l2 + 1 <= n) ? l2 + 1 : n;

            i32 l1_idx = l1 - 1;
            i32 l2_idx = l2 - 1;
            i32 l2p1_idx = l2p1 - 1;

            if (discr) {
                i32 len = n - l2;
                g11 = (len > 0) ? -SLC_DDOT(&len, &s[l1_idx + l2p1_idx*lds], &lds, &c[l2p1_idx], &one_inc) : zero;

                if (tbyt) {
                    g12 = (len > 0) ? -SLC_DDOT(&len, &s[l1_idx + l2p1_idx*lds], &lds, &c[l2p1_idx + ldc], &one_inc) : zero;
                    vec[0] = c[l1_idx + 0*ldc] + g11*at[0] + g12*at[1];
                    vec[2] = c[l1_idx + 1*ldc] + g11*at[2] + g12*at[3];
                } else {
                    vec[0] = c[l1_idx + 0*ldc] + g11*at[0];
                }

                if (dl != 1) {
                    g21 = (len > 0) ? -SLC_DDOT(&len, &s[l2_idx + l2p1_idx*lds], &lds, &c[l2p1_idx], &one_inc) : zero;

                    if (tbyt) {
                        g22 = (len > 0) ? -SLC_DDOT(&len, &s[l2_idx + l2p1_idx*lds], &lds, &c[l2p1_idx + ldc], &one_inc) : zero;
                        vec[1] = c[l2_idx + 0*ldc] + g21*at[0] + g22*at[1];
                        vec[3] = c[l2_idx + 1*ldc] + g21*at[2] + g22*at[3];
                    } else {
                        vec[1] = c[l2_idx + 0*ldc] + g21*at[0];
                    }
                }

                i32 info_local;
                sb04px(false, false, -isgn, dl, m,
                       &s[l1_idx + l1_idx*lds], lds, at, 2,
                       vec, 2, &scaloc, x, 2, &xnorm, &info_local);
                if (info_local > infom) infom = info_local;
            } else {
                i32 len = n - l2;
                vec[0] = c[l1_idx + 0*ldc] - ((len > 0) ? SLC_DDOT(&len, &s[l1_idx + l2p1_idx*lds], &lds, &c[l2p1_idx], &one_inc) : zero);

                if (tbyt) {
                    vec[2] = c[l1_idx + 1*ldc] - ((len > 0) ? SLC_DDOT(&len, &s[l1_idx + l2p1_idx*lds], &lds, &c[l2p1_idx + ldc], &one_inc) : zero);
                }

                if (dl != 1) {
                    vec[1] = c[l2_idx + 0*ldc] - ((len > 0) ? SLC_DDOT(&len, &s[l2_idx + l2p1_idx*lds], &lds, &c[l2p1_idx], &one_inc) : zero);

                    if (tbyt) {
                        vec[3] = c[l2_idx + 1*ldc] - ((len > 0) ? SLC_DDOT(&len, &s[l2_idx + l2p1_idx*lds], &lds, &c[l2p1_idx + ldc], &one_inc) : zero);
                    }
                }

                i32 info_local;
                i32 ltranl = 0;
                i32 ltranr = 0;
                SLC_DLASY2(&ltranl, &ltranr, &isgn, &dl, &m,
                           &s[l1_idx + l1_idx*lds], &lds, at, (i32[]){2},
                           vec, (i32[]){2}, &scaloc, x, (i32[]){2}, &xnorm, &info_local);
                if (info_local > infom) infom = info_local;
            }

            if (scaloc != one) {
                for (i32 jj = 0; jj < m; jj++) {
                    i32 n_local = n;
                    SLC_DSCAL(&n_local, &scaloc, &c[jj*ldc], &one_inc);
                }
                *scale = (*scale) * scaloc;
            }

            c[l1_idx + 0*ldc] = x[0];
            if (tbyt) c[l1_idx + 1*ldc] = x[2];
            if (dl != 1) {
                c[l2_idx + 0*ldc] = x[1];
                if (tbyt) c[l2_idx + 1*ldc] = x[3];
            }
        }
    } else {
        lnext = 1;

        for (l = 1; l <= n; l++) {
            if (l < lnext) {
                continue;
            }
            l1 = l;
            l2 = l;
            if (l < n) {
                if (s[l + (l-1)*lds] != zero) {
                    l2 = l2 + 1;
                }
                lnext = l2 + 1;
            }
            dl = l2 - l1 + 1;

            i32 l1_idx = l1 - 1;
            i32 l2_idx = l2 - 1;

            if (discr) {
                i32 len = l1 - 1;
                g11 = (len > 0) ? -SLC_DDOT(&len, &c[0], &one_inc, &s[l1_idx*lds], &one_inc) : zero;

                if (tbyt) {
                    g21 = (len > 0) ? -SLC_DDOT(&len, &c[ldc], &one_inc, &s[l1_idx*lds], &one_inc) : zero;
                    vec[0] = c[l1_idx + 0*ldc] + at[0]*g11 + at[2]*g21;
                    vec[1] = c[l1_idx + 1*ldc] + at[1]*g11 + at[3]*g21;
                } else {
                    vec[0] = c[l1_idx + 0*ldc] + at[0]*g11;
                }

                if (dl != 1) {
                    g12 = (len > 0) ? -SLC_DDOT(&len, &c[0], &one_inc, &s[l2_idx*lds], &one_inc) : zero;

                    if (tbyt) {
                        g22 = (len > 0) ? -SLC_DDOT(&len, &c[ldc], &one_inc, &s[l2_idx*lds], &one_inc) : zero;
                        vec[2] = c[l2_idx + 0*ldc] + at[0]*g12 + at[2]*g22;
                        vec[3] = c[l2_idx + 1*ldc] + at[1]*g12 + at[3]*g22;
                    } else {
                        vec[2] = c[l2_idx + 0*ldc] + at[0]*g12;
                    }
                }

                i32 info_local;
                sb04px(false, false, -isgn, m, dl,
                       at, 2, &s[l1_idx + l1_idx*lds], lds,
                       vec, 2, &scaloc, x, 2, &xnorm, &info_local);
                if (info_local > infom) infom = info_local;
            } else {
                i32 len = l1 - 1;
                vec[0] = c[l1_idx + 0*ldc] - ((len > 0) ? SLC_DDOT(&len, &c[0], &one_inc, &s[l1_idx*lds], &one_inc) : zero);

                if (tbyt) {
                    vec[1] = c[l1_idx + 1*ldc] - ((len > 0) ? SLC_DDOT(&len, &c[ldc], &one_inc, &s[l1_idx*lds], &one_inc) : zero);
                }

                if (dl != 1) {
                    vec[2] = c[l2_idx + 0*ldc] - ((len > 0) ? SLC_DDOT(&len, &c[0], &one_inc, &s[l2_idx*lds], &one_inc) : zero);

                    if (tbyt) {
                        vec[3] = c[l2_idx + 1*ldc] - ((len > 0) ? SLC_DDOT(&len, &c[ldc], &one_inc, &s[l2_idx*lds], &one_inc) : zero);
                    }
                }

                i32 info_local;
                i32 ltranl = 0;
                i32 ltranr = 0;
                SLC_DLASY2(&ltranl, &ltranr, &isgn, &m, &dl,
                           at, (i32[]){2}, &s[l1_idx + l1_idx*lds], &lds,
                           vec, (i32[]){2}, &scaloc, x, (i32[]){2}, &xnorm, &info_local);
                if (info_local > infom) infom = info_local;
            }

            if (scaloc != one) {
                for (i32 jj = 0; jj < m; jj++) {
                    i32 n_local = n;
                    SLC_DSCAL(&n_local, &scaloc, &c[jj*ldc], &one_inc);
                }
                *scale = (*scale) * scaloc;
            }

            c[l1_idx + 0*ldc] = x[0];
            if (tbyt) c[l1_idx + 1*ldc] = x[1];
            if (dl != 1) {
                c[l2_idx + 0*ldc] = x[2];
                if (tbyt) c[l2_idx + 1*ldc] = x[3];
            }
        }
    }

    *info = infom;
}
