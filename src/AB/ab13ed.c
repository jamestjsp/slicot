/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

void ab13ed(i32 n, f64 *a, i32 lda, f64 *low, f64 *high, f64 tol, f64 *dwork, i32 ldwork, i32 *info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    i32 i, ia2, iaa, igf, ihi, ilo, iwi, iwk, iwr, jwork, minwrk, n2;
    f64 anrm, seps, sfmn, sigma, tau, temp, tol1, tol2;
    bool rneg, sufwrk;

    *info = 0;
    minwrk = 3 * n * (n + 1);

    if (n < 0) {
        *info = -1;
    } else if (lda < MAX(1, n)) {
        *info = -3;
    } else if (ldwork < MAX(1, minwrk)) {
        *info = -8;
    }

    if (*info != 0) {
        i32 info_val = -(*info);
        SLC_XERBLA("AB13ED", &info_val);
        return;
    }

    *low = zero;
    if (n == 0) {
        *high = zero;
        dwork[0] = one;
        return;
    }

    n2 = n * n;
    igf = 0;
    ia2 = igf + n2 + n;
    iaa = ia2 + n2;
    iwk = iaa + n2;
    iwr = iaa;
    iwi = iwr + n;

    sufwrk = (ldwork - iwk >= n2);

    sfmn = SLC_DLAMCH("Safe minimum");
    seps = sqrt(SLC_DLAMCH("Epsilon"));
    tau = one + MAX(tol, seps);

    char frob = 'F';
    anrm = SLC_DLANGE(&frob, &n, &n, a, &lda, dwork);

    tol1 = seps * anrm;
    tol2 = tol1 * (f64)(2 * n);

    *high = anrm;

    while (*high > (tau * MAX(tol1, *low))) {
        sigma = sqrt(*high) * sqrt(MAX(tol1, *low));

        SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[iaa], &n);

        dwork[igf] = sigma;
        dwork[igf + n] = -sigma;

        for (i = 1; i < n; i++) {
            dwork[igf + i] = zero;
        }

        i32 c_limit = ia2 - n - 2;
        i32 inc = 1;
        i32 len_copy = n + 1;
        i32 step = n + 1;
        for (i = igf; i <= c_limit; i += step) {
            SLC_DCOPY(&len_copy, &dwork[i], &inc, &dwork[i + n + 1], &inc);
        }

        f64 dummy2;
        i32 one_int = 1;
        mb04zd("No vectors", n, &dwork[iaa], n, &dwork[igf], n, &dummy2, one_int, &dwork[iwk], info);

        jwork = ia2;
        if (sufwrk) {
            jwork = iwk;
        }

        SLC_DLACPY("Lower", &n, &n, &dwork[igf], &n, &dwork[jwork], &n);
        ma02ed('L', n, &dwork[jwork], n);

        if (sufwrk) {
            SLC_DSYMM("Left", "Upper", &n, &n, &one, &dwork[igf + n], &n, &dwork[jwork], &n, &zero, &dwork[ia2], &n);
        } else {
            for (i = 1; i <= n; i++) {
                SLC_DSYMV("Upper", &n, &one, &dwork[igf + n], &n, &dwork[ia2 + n * (i - 1)], &one_int, &zero, &dwork[iwk], &one_int);
                SLC_DCOPY(&n, &dwork[iwk], &one_int, &dwork[ia2 + n * (i - 1)], &one_int);
            }
        }

        SLC_DGEMM("NoTranspose", "NoTranspose", &n, &n, &n, &one, &dwork[iaa], &n, &dwork[iaa], &n, &one, &dwork[ia2], &n);

        jwork = iwi + n;
        i32 i_info;
        SLC_DGEBAL("Scale", &n, &dwork[ia2], &n, &ilo, &ihi, &dwork[jwork], &i_info);

        SLC_DHSEQR("Eigenvalues", "NoSchurVectors", &n, &ilo, &ihi, &dwork[ia2], &n, &dwork[iwr], &dwork[iwi], &dummy2, &one_int, &dwork[jwork], &n, info);

        if (*info != 0) {
            *info = 1;
            return;
        }

        i = 0;
        rneg = false;

        while (!rneg && i < n) {
            temp = fabs(dwork[iwi + i]);
            if (tol1 > sfmn) temp = temp / tol1;

            rneg = ((dwork[iwr + i] < zero) && (temp <= tol2));
            i++;
        }

        if (rneg) {
            *high = sigma;
        } else {
            *low = sigma;
        }
    }

    dwork[0] = (f64)MAX(4 * n2 + n, minwrk);
}
