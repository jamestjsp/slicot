// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"

void mb03vd(i32 n, i32 p, i32 ilo, i32 ihi, f64* a, i32 lda1, i32 lda2,
            f64* tau, i32 ldtau, f64* dwork, i32* info) {
    const f64 zero = 0.0;
    i32 int1 = 1;
    i32 int0 = 0;

    *info = 0;

    i32 max_1_n = (1 > n) ? 1 : n;
    i32 min_ilo_n = (ilo < n) ? ilo : n;

    if (n < 0) {
        *info = -1;
    } else if (p < 1) {
        *info = -2;
    } else if (ilo < 1 || ilo > max_1_n) {
        *info = -3;
    } else if (ihi < min_ilo_n || ihi > n) {
        *info = -4;
    } else if (lda1 < max_1_n) {
        *info = -6;
    } else if (lda2 < max_1_n) {
        *info = -7;
    } else if (ldtau < (n > 1 ? n - 1 : 1)) {
        *info = -9;
    }

    if (*info != 0) {
        return;
    }

    i32 nh = ihi - ilo + 1;
    if (nh <= 1) {
        return;
    }

    f64 dummy = zero;
    i32 lda12 = lda1 * lda2;

    for (i32 i = ilo - 1; i < ihi - 1; i++) {
        i32 i1 = i + 1;
        i32 i2 = (i + 2 < n) ? (i + 2) : (n - 1);

        for (i32 j = p - 1; j >= 1; j--) {
            f64* aj = a + j * lda12;
            f64* aj_prev = a + (j - 1) * lda12;
            f64* tau_j = tau + j * ldtau;

            for (i32 k = 0; k < ilo - 1; k++) {
                tau_j[k] = zero;
            }
            if (ihi < n) {
                i32 n_copy = n - ihi;
                SLC_DCOPY(&n_copy, &dummy, &int0, &tau_j[ihi - 1], &int1);
            }

            i32 len = ihi - i;
            SLC_DLARFG(&len, &aj[i + i * lda1], &aj[i1 + i * lda1], &int1, &tau_j[i]);

            SLC_MB04PY('R', ihi, len, &aj[i1 + i * lda1], tau_j[i],
                       &aj_prev[0 + i * lda1], lda1, dwork);

            i32 ncols = n - i - 1;
            SLC_MB04PY('L', len, ncols, &aj[i1 + i * lda1], tau_j[i],
                       &aj[i + i1 * lda1], lda1, dwork);
        }

        f64* a1 = a;
        f64* ap = a + (p - 1) * lda12;
        f64* tau_1 = tau;

        i32 len1 = ihi - i - 1;
        SLC_DLARFG(&len1, &a1[i1 + i * lda1], &a1[i2 + i * lda1], &int1, &tau_1[i]);

        i32 ncols_r = ihi - i - 1;
        SLC_MB04PY('R', ihi, ncols_r, &a1[i2 + i * lda1], tau_1[i],
                   &ap[0 + i1 * lda1], lda1, dwork);

        i32 ncols_l = n - i - 1;
        SLC_MB04PY('L', len1, ncols_l, &a1[i2 + i * lda1], tau_1[i],
                   &a1[i1 + i1 * lda1], lda1, dwork);
    }
}
