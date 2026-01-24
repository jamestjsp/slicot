/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB02WD - Conjugate gradient solver for SPD linear systems
 *
 * Solves Ax = b for symmetric positive definite A using CG algorithm,
 * or f(A, x) = b for implicit function form.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void mb02wd(
    const char* form,
    mb02wd_func f,
    const i32 n,
    i32* ipar,
    const i32 lipar,
    f64* dpar,
    const i32 ldpar,
    const i32 itmax,
    f64* a,
    const i32 lda,
    const f64* b,
    const i32 incb,
    f64* x,
    const i32 incx,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 NEG_ONE = -1.0;
    const i32 int1 = 1;

    char form_c = toupper((unsigned char)form[0]);
    bool mat = (form_c == 'U' || form_c == 'L');

    *iwarn = 0;
    *info = 0;

    if (!mat && form_c != 'F') {
        *info = -1;
    } else if (n < 0) {
        *info = -3;
    } else if (!mat && lipar < 0) {
        *info = -5;
    } else if (!mat && ldpar < 0) {
        *info = -7;
    } else if (itmax < 0) {
        *info = -8;
    } else if (lda < 1 || (mat && lda < n)) {
        *info = -10;
    } else if (incb <= 0) {
        *info = -12;
    } else if (incx <= 0) {
        *info = -14;
    } else if (ldwork < (2 > 3*n ? 2 : 3*n)) {
        *info = -17;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = ZERO;
        dwork[1] = ZERO;
        return;
    }

    if (itmax == 0) {
        dwork[0] = ZERO;
        *iwarn = 2;
        return;
    }

    f64 toldef = tol;
    if (toldef <= ZERO) {
        f64 bnorm = SLC_DNRM2(&n, b, &incb);
        toldef = (f64)n * SLC_DLAMCH("Epsilon") * bnorm;
    }

    i32 k = 0;

    i32 aq = n;
    i32 r = n + aq;
    i32 dwleft = n + r;

    if (mat) {
        SLC_DCOPY(&n, b, &incb, &dwork[r], &int1);
        SLC_DSYMV(&form_c, &n, &ONE, a, &lda, x, &incx, &NEG_ONE, &dwork[r], &int1);
    } else {
        SLC_DCOPY(&n, x, &incx, &dwork[r], &int1);
        i32 ldwork_f = ldwork - dwleft;
        f(n, ipar, lipar, dpar, ldpar, a, lda, &dwork[r], 1, &dwork[dwleft], ldwork_f, info);
        if (*info != 0) {
            return;
        }
        SLC_DAXPY(&n, &NEG_ONE, b, &incb, &dwork[r], &int1);
    }

    SLC_DCOPY(&n, &dwork[r], &int1, dwork, &int1);

    f64 res = SLC_DNRM2(&n, &dwork[r], &int1);

    if (res <= toldef) {
        goto finish;
    }

    while (k < itmax) {
        if (mat) {
            SLC_DSYMV(&form_c, &n, &ONE, a, &lda, dwork, &int1, &ZERO, &dwork[aq], &int1);
        } else {
            SLC_DCOPY(&n, dwork, &int1, &dwork[aq], &int1);
            i32 ldwork_f = ldwork - dwleft;
            f(n, ipar, lipar, dpar, ldpar, a, lda, &dwork[aq], 1, &dwork[dwleft], ldwork_f, info);
            if (*info != 0) {
                return;
            }
        }

        f64 qr = SLC_DDOT(&n, dwork, &int1, &dwork[r], &int1);
        f64 qaq = SLC_DDOT(&n, dwork, &int1, &dwork[aq], &int1);
        f64 alpha = qr / qaq;

        f64 neg_alpha = -alpha;
        SLC_DAXPY(&n, &neg_alpha, dwork, &int1, x, &incx);

        SLC_DAXPY(&n, &neg_alpha, &dwork[aq], &int1, &dwork[r], &int1);

        f64 resold = res;
        res = SLC_DNRM2(&n, &dwork[r], &int1);

        if (res <= toldef) {
            goto finish;
        }

        f64 beta = (res / resold) * (res / resold);

        SLC_DSCAL(&n, &beta, dwork, &int1);
        SLC_DAXPY(&n, &ONE, &dwork[r], &int1, dwork, &int1);

        k++;
    }

    *iwarn = 1;

finish:
    dwork[0] = (f64)k;
    dwork[1] = res;
}
