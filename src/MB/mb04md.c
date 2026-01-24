/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

/**
 * @brief Balances a general real matrix to reduce its 1-norm.
 *
 * @details
 * To reduce the 1-norm of a general real matrix A by balancing.
 * This involves diagonal similarity transformations applied
 * iteratively to A to make the rows and columns as close in norm as
 * possible.
 *
 * This routine can be used instead LAPACK Library routine DGEBAL,
 * when no reduction of the 1-norm of the matrix is possible with
 * DGEBAL, as for upper triangular matrices.
 *
 * @param[in] n
 *     The order of the matrix A.  N >= 0.
 *
 * @param[in,out] maxred
 *     On entry, the maximum allowed reduction in the 1-norm of
 *     A (in an iteration) if zero rows or columns are encountered.
 *     If MAXRED > 0.0, MAXRED must be larger than one.
 *     If MAXRED <= 0.0, then the value 10.0 is used.
 *     On exit, if the 1-norm of the given matrix A is non-zero,
 *     the ratio between the 1-norm of the given matrix and the
 *     1-norm of the balanced matrix.
 *
 * @param[in,out] a
 *     On entry, the leading N-by-N part contains the input matrix A.
 *     On exit, the leading N-by-N part contains the balanced matrix.
 *
 * @param[in] lda
 *     The leading dimension of the array A.  LDA >= max(1,N).
 *
 * @param[out] scale
 *     The scaling factors applied to A. If D(j) is the scaling
 *     factor applied to row and column j, then SCALE(j) = D(j),
 *     for j = 1,...,N.
 *
 * @param[out] info
 *     = 0:  successful exit.
 *     < 0:  if INFO = -i, the i-th argument had an illegal value.
 */
void mb04md(const i32 n, f64 *maxred, f64 *a, const i32 lda, f64 *scale, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 SCLFAC = 10.0;
    const f64 FACTOR = 0.95;
    const f64 MAXR = 10.0;

    bool noconv;
    i32 i, j, ica, ira;
    f64 anorm, c, ca, f, g, r, ra, s, sred, maxnrm;
    f64 sfmax1, sfmax2, sfmin1, sfmin2;
    i32 int1 = 1;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (*maxred > ZERO && *maxred < ONE) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("MB04MD", &neg_info);
        return;
    }

    if (n == 0) {
        return;
    }

    for (i = 0; i < n; i++) {
        scale[i] = ONE;
    }

    anorm = SLC_DLANGE("1", &n, &n, a, &lda, scale);
    if (anorm == ZERO) {
        return;
    }

    sfmin1 = SLC_DLAMCH("S") / SLC_DLAMCH("P");
    sfmax1 = ONE / sfmin1;
    sfmin2 = sfmin1 * SCLFAC;
    sfmax2 = ONE / sfmin2;

    sred = *maxred;
    if (sred <= ZERO) {
        sred = MAXR;
    }

    maxnrm = anorm / sred;
    if (maxnrm < sfmin1) {
        maxnrm = sfmin1;
    }

    do {
        noconv = false;

        for (i = 0; i < n; i++) {
            c = ZERO;
            r = ZERO;

            for (j = 0; j < n; j++) {
                if (j == i) {
                    continue;
                }
                c += fabs(a[j + i * lda]);
                r += fabs(a[i + j * lda]);
            }

            ica = SLC_IDAMAX(&n, &a[i * lda], &int1) - 1;
            ca = fabs(a[ica + i * lda]);
            ira = SLC_IDAMAX(&n, &a[i], &lda) - 1;
            ra = fabs(a[i + ira * lda]);

            if (c == ZERO && r == ZERO) {
                continue;
            }
            if (c == ZERO) {
                if (r <= maxnrm) {
                    continue;
                }
                c = maxnrm;
            }
            if (r == ZERO) {
                if (c <= maxnrm) {
                    continue;
                }
                r = maxnrm;
            }

            g = r / SCLFAC;
            f = ONE;
            s = c + r;

            while (c < g && fmax(f, fmax(c, ca)) < sfmax2 && fmin(r, fmin(g, ra)) > sfmin2) {
                f *= SCLFAC;
                c *= SCLFAC;
                ca *= SCLFAC;
                r /= SCLFAC;
                g /= SCLFAC;
                ra /= SCLFAC;
            }

            g = c / SCLFAC;

            while (g >= r && fmax(r, ra) < sfmax2 && fmin(f, fmin(c, fmin(g, ca))) > sfmin2) {
                f /= SCLFAC;
                c /= SCLFAC;
                g /= SCLFAC;
                ca /= SCLFAC;
                r *= SCLFAC;
                ra *= SCLFAC;
            }

            if ((c + r) >= FACTOR * s) {
                continue;
            }
            if (f < ONE && scale[i] < ONE) {
                if (f * scale[i] <= sfmin1) {
                    continue;
                }
            }
            if (f > ONE && scale[i] > ONE) {
                if (scale[i] >= sfmax1 / f) {
                    continue;
                }
            }

            g = ONE / f;
            scale[i] *= f;
            noconv = true;

            SLC_DSCAL(&n, &g, &a[i], &lda);
            SLC_DSCAL(&n, &f, &a[i * lda], &int1);
        }
    } while (noconv);

    f64 bnorm = SLC_DLANGE("1", &n, &n, a, &lda, scale);
    *maxred = anorm / bnorm;
}
