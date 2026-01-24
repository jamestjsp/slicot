// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdlib.h>

void mb03sd(const char *jobscl, i32 n, f64 *a, i32 lda, f64 *qg, i32 ldqg,
            f64 *wr, f64 *wi, f64 *dwork, i32 ldwork, i32 *info) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    i32 n2 = n * n;
    i32 int1 = 1;
    i32 ignore;

    char jobscl_upper = toupper((unsigned char)jobscl[0]);
    bool scale = (jobscl_upper == 'S');

    *info = 0;

    if (!scale && jobscl_upper != 'N') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ldqg < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldwork < (n2 + n > 1 ? n2 + n : 1)) {
        *info = -10;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    i32 chunk = (ldwork - n2) / n;
    bool block = (chunk < n ? chunk : n) > 1;
    bool blas3 = chunk >= n;

    i32 jwork;
    if (blas3) {
        jwork = n2;  // 0-based offset
    } else {
        jwork = 0;
    }

    // Copy Q' (lower triangle) from QG to workspace and symmetrize
    // DLACPY('Lower', N, N, QG, LDQG, DWORK(JWORK), N)
    SLC_DLACPY("Lower", &n, &n, qg, &ldqg, &dwork[jwork], &n);

    // MA02ED('Lower', N, DWORK(JWORK), N) - symmetrize from lower triangle
    ma02ed('L', n, &dwork[jwork], n);

    if (blas3) {
        // DSYMM('Left', 'Upper', N, N, ONE, QG(1,2), LDQG, DWORK(JWORK), N, ZERO, DWORK, N)
        // G' is in upper triangle of columns 2:N+1 of QG
        SLC_DSYMM("Left", "Upper", &n, &n, &ONE, &qg[0 + 1*ldqg], &ldqg,
                  &dwork[jwork], &n, &ZERO, dwork, &n);
    } else if (block) {
        i32 jw = n2;  // 0-based offset

        for (i32 j = 0; j < n; j += chunk) {
            i32 bl = n - j;
            if (bl > chunk) bl = chunk;

            SLC_DSYMM("Left", "Upper", &n, &bl, &ONE, &qg[0 + 1*ldqg], &ldqg,
                      &dwork[n * j], &n, &ZERO, &dwork[jw], &n);
            SLC_DLACPY("Full", &n, &bl, &dwork[jw], &n, &dwork[n * j], &n);
        }
    } else {
        // BLAS 2 calculation
        for (i32 j = 0; j < n; j++) {
            SLC_DSYMV("Upper", &n, &ONE, &qg[0 + 1*ldqg], &ldqg,
                      &dwork[n * j], &int1, &ZERO, wr, &int1);
            SLC_DCOPY(&n, wr, &int1, &dwork[n * j], &int1);
        }
    }

    // DGEMM('N', 'N', N, N, N, ONE, A, LDA, A, LDA, ONE, DWORK, N)
    // A'' = A'^2 + G'*Q'
    SLC_DGEMM("N", "N", &n, &n, &n, &ONE, a, &lda, a, &lda, &ONE, dwork, &n);

    // Clear lower part for scaling if N > 2
    if (scale && n > 2) {
        i32 nm2 = n - 2;
        SLC_DLASET("Lower", &nm2, &nm2, &ZERO, &ZERO, &dwork[2], &n);
    }

    // Balance and compute eigenvalues
    i32 ilo, ihi;
    SLC_DGEBAL(jobscl, &n, dwork, &n, &ilo, &ihi, &dwork[n2], &ignore);

    f64 dummy[1];
    i32 ldummy = 1;
    SLC_DHSEQR("E", "N", &n, &ilo, &ihi, dwork, &n, wr, wi, dummy, &ldummy,
               &dwork[n2], &n, info);

    if (*info == 0) {
        // Compute square roots of eigenvalues
        for (i32 i = 0; i < n; i++) {
            f64 x = wr[i];
            f64 y = wi[i];
            ma01ad(x, y, &wr[i], &wi[i]);
        }

        // Bubble sort: decreasing real part, then decreasing imaginary for zero real
        bool sorted = false;
        for (i32 m = n; m >= 1 && !sorted; m--) {
            sorted = true;
            for (i32 i = 0; i < m - 1; i++) {
                bool swap_needed = false;
                if (wr[i] < wr[i+1]) {
                    swap_needed = true;
                } else if (wr[i] == ZERO && wr[i+1] == ZERO && wi[i] < wi[i+1]) {
                    swap_needed = true;
                }

                if (swap_needed) {
                    f64 swap = wr[i];
                    wr[i] = wr[i+1];
                    wr[i+1] = swap;

                    swap = wi[i];
                    wi[i] = wi[i+1];
                    wi[i+1] = swap;

                    sorted = false;
                }
            }
        }
    }

    dwork[0] = (f64)(2 * n2);
}
