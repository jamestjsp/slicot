/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

void mb05od(
    const char* balanc,
    const i32 n,
    const i32 ndiag,
    const f64 delta,
    f64* a,
    const i32 lda,
    i32* mdig,
    i32* idig,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 FOUR = 4.0;
    const f64 EIGHT = 8.0;
    const f64 TEN = 10.0;
    const f64 TWELVE = 12.0;
    const f64 NINTEN = 19.0;
    const f64 TWO4 = 24.0;
    const f64 FOUR7 = 47.0;
    const f64 TWOHND = 200.0;

    *iwarn = 0;
    *info = 0;

    bool lbals = (balanc[0] == 'S' || balanc[0] == 's');

    if (!(balanc[0] == 'N' || balanc[0] == 'n' || lbals)) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ndiag < 1) {
        *info = -3;
    } else if (lda < ((n > 1) ? n : 1)) {
        *info = -6;
    } else if ((ldwork < 1) ||
               (ldwork < n * (2 * n + ndiag + 1) + ndiag && n > 1)) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    f64 eps = SLC_DLAMCH("Epsilon");
    i32 ndec = (i32)(log10(ONE / eps) + ONE);

    if (n == 0) {
        *mdig = ndec;
        *idig = ndec;
        return;
    }

    i32 base = (i32)SLC_DLAMCH("Base");
    i32 ndecm1 = ndec - 1;
    f64 underf = SLC_DLAMCH("Underflow");
    f64 ovrthr = SLC_DLAMCH("Overflow");
    f64 ovrth2 = sqrt(ovrthr);

    if (delta == ZERO) {
        SLC_DLASET("Full", &n, &n, &ZERO, &ONE, a, &lda);
        *mdig = ndecm1;
        *idig = ndecm1;
        return;
    }

    if (n == 1) {
        a[0] = exp(a[0] * delta);
        *mdig = ndecm1;
        *idig = ndecm1;
        return;
    }

    i32 jwora1 = 0;
    i32 jwora2 = jwora1 + n * n;
    i32 jwora3 = jwora2 + n * ndiag;
    i32 jworv1 = jwora3 + n * n;
    i32 jworv2 = jworv1 + n;

    dwork[jworv2] = HALF;

    for (i32 i = 1; i < ndiag; i++) {
        i32 im1 = i;
        dwork[jworv2 + i] = dwork[jworv2 + i - 1] * (f64)(ndiag - im1) /
                            (f64)((i + 1) * (2 * ndiag - im1));
    }

    f64 vareps = eps * eps * (((f64)base * (f64)base - ONE) /
                 (TWO4 * log((f64)base)));
    f64 xn = (f64)n;
    f64 tr = ZERO;

    i32 inc1 = 1;
    for (i32 i = 0; i < n; i++) {
        SLC_DSCAL(&n, &delta, &a[i * lda], &inc1);
        tr += a[i + i * lda];
    }

    f64 avgev = tr / xn;
    if (avgev > log(ovrthr) || avgev < log(underf)) {
        avgev = ZERO;
    }

    f64 anorm;
    if (avgev != ZERO) {
        anorm = SLC_DLANGE("1-norm", &n, &n, a, &lda, &dwork[jwora1]);

        for (i32 i = 0; i < n; i++) {
            a[i + i * lda] -= avgev;
        }

        f64 temp = SLC_DLANGE("1-norm", &n, &n, a, &lda, &dwork[jwora1]);
        if (temp > HALF * anorm) {
            for (i32 i = 0; i < n; i++) {
                a[i + i * lda] += avgev;
            }
            avgev = ZERO;
        }
    }

    char actbal = balanc[0];
    if (lbals) {
        SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[jwora1], &n);
        f64 maxred = TWOHND;
        mb04md(n, &maxred, a, lda, &dwork[jworv1], info);
        if (maxred < ONE) {
            SLC_DLACPY("Full", &n, &n, &dwork[jwora1], &n, a, &lda);
            actbal = 'N';
            dwork[jworv1] = ONE;
            i32 nm1 = n - 1;
            SLC_DCOPY(&nm1, &dwork[jworv1], &(i32){0}, &dwork[jworv1 + 1], &inc1);
            *iwarn = 3;
        }
    }

    anorm = SLC_DLANGE("1-norm", &n, &n, a, &lda, &dwork[jwora1]);
    i32 m = 0;
    f64 factor;

    if (anorm >= HALF) {
        i32 mpower = (i32)(log(ovrthr) / log(TWO));
        m = (i32)(log(anorm) / log(TWO)) + 1;
        if (m > mpower) {
            *info = 1;
            return;
        }
        factor = pow(TWO, (f64)m);
        if (m + 1 < mpower) {
            m = m + 1;
            factor = factor * TWO;
        }

        f64 inv_factor = ONE / factor;
        for (i32 i = 0; i < n; i++) {
            SLC_DSCAL(&n, &inv_factor, &a[i * lda], &inc1);
        }
    }

    i32 ndagm1 = ndiag - 1;
    i32 ndagm2 = ndagm1 - 1;
    i32 ij = 0;

    for (i32 j = 0; j < n; j++) {
        SLC_DGEMV("No transpose", &n, &n, &ONE, a, &lda, &a[j * lda], &inc1,
                  &ZERO, &dwork[jwora2], &inc1);
        i32 ik = 0;

        for (i32 k = 0; k < ndagm2; k++) {
            SLC_DGEMV("No transpose", &n, &n, &ONE, a, &lda,
                      &dwork[jwora2 + ik], &inc1, &ZERO, &dwork[jwora2 + ik + n],
                      &inc1);
            ik += n;
        }

        for (i32 i = 0; i < n; i++) {
            f64 s = ZERO;
            f64 u = ZERO;
            ik = ndagm2 * n + i;

            for (i32 k = ndagm1; k >= 1; k--) {
                f64 p = dwork[jworv2 + k] * dwork[jwora2 + ik];
                ik -= n;
                s += p;
                if ((k + 1) % 2 == 0) {
                    u += p;
                } else {
                    u -= p;
                }
            }

            f64 p = dwork[jworv2] * a[i + j * lda];
            s += p;
            u -= p;
            if (i == j) {
                s += ONE;
                u += ONE;
            }
            dwork[jwora3 + ij] = s;
            dwork[jwora1 + ij] = u;
            ij++;
        }
    }

    i32 ifail;
    SLC_DGETRF(&n, &n, &dwork[jwora1], &n, iwork, &ifail);
    if (ifail > 0) {
        *info = 2;
        return;
    }

    SLC_DLACPY("Full", &n, &n, &dwork[jwora3], &n, a, &lda);
    SLC_DGETRS("No transpose", &n, &n, &dwork[jwora1], &n, iwork, a,
               &lda, &ifail);

    anorm = SLC_DLANGE("1-norm", &n, &n, a, &lda, &dwork[jwora1]);
    f64 eabs;
    if (anorm >= ONE) {
        eabs = (NINTEN * xn + FOUR7) * (eps * anorm);
    } else {
        eabs = ((NINTEN * xn + FOUR7) * eps) * anorm;
    }

    f64 var;
    if (m != 0) {
        var = xn * vareps;
        f64 fn = (FOUR * xn) / ((xn + TWO) * (xn + ONE));
        f64 gn = ((TWO * xn + TEN) * xn - FOUR) /
                 (((xn + TWO) * (xn + TWO)) * ((xn + ONE) * (xn + ONE)));

        for (i32 k = 0; k < m; k++) {
            f64 temp;
            if (anorm > ovrth2) {
                f64 inv_anorm = ONE / anorm;
                SLC_DGEMM("No transpose", "No transpose", &n, &n, &n,
                          &inv_anorm, a, &lda, a, &lda, &ZERO,
                          &dwork[jwora1], &n);
                f64 s = SLC_DLANGE("1-norm", &n, &n, &dwork[jwora1], &n,
                                   &dwork[jwora1]);
                if (anorm <= ovrthr / s) {
                    SLC_DLASCL("General", &n, &n, &ONE, &anorm, &n, &n,
                               &dwork[jwora1], &n, info);
                    temp = ovrthr;
                } else {
                    *info = 3;
                    return;
                }
            } else {
                SLC_DGEMM("No transpose", "No transpose", &n, &n, &n, &ONE,
                          a, &lda, a, &lda, &ZERO, &dwork[jwora1], &n);
                temp = anorm * anorm;
            }

            if (eabs < ONE) {
                eabs = (TWO * anorm + eabs) * eabs + xn * (eps * temp);
            } else if (eabs < sqrt(ONE - xn * eps + ovrthr / temp) * anorm -
                                anorm) {
                eabs = xn * (eps * temp) + TWO * (anorm * eabs) + eabs * eabs;
            } else {
                eabs = ovrthr;
            }

            f64 tmp1 = fn * var + gn * (temp * vareps);
            if (tmp1 > ovrthr / temp) {
                var = ovrthr;
            } else {
                var = tmp1 * temp;
            }

            SLC_DLACPY("Full", &n, &n, &dwork[jwora1], &n, a, &lda);
            anorm = SLC_DLANGE("1-norm", &n, &n, a, &lda, &dwork[jwora1]);
        }
    } else {
        var = (TWELVE * xn) * vareps;
    }

    mb05oy(&actbal, n, 1, n, a, lda, &dwork[jworv1], info);
    f64 eavgev = exp(avgev);
    f64 emnorm = SLC_DLANGE("1-norm", &n, &n, a, &lda, &dwork[jwora1]);

    f64 big = ONE;
    f64 small = ONE;
    f64 sum2d;

    if (lbals) {
        for (i32 i = 0; i < n; i++) {
            f64 u = dwork[jworv1 + i];
            if (big < u) big = u;
            if (small > u) small = u;
        }
        sum2d = SLC_DNRM2(&n, &dwork[jworv1], &inc1);
    } else {
        sum2d = sqrt(xn);
    }

    f64 sd2 = sqrt(EIGHT * xn * vareps) * anorm;
    f64 bd = sqrt(var);
    f64 ss = (bd > sd2) ? bd : sd2;
    bd = (bd < sd2) ? bd : sd2;
    sd2 = ss * sqrt(ONE + (bd / ss) * (bd / ss));

    if (sd2 <= ONE) {
        sd2 = (TWO / xn) * sum2d * sd2;
    } else if (sum2d / xn < ovrthr / TWO / sd2) {
        sd2 = (TWO / xn) * sum2d * sd2;
    } else {
        sd2 = ovrthr;
    }

    f64 size;
    if (lbals) {
        size = ZERO;
    } else {
        if (sd2 < ovrthr - emnorm) {
            size = emnorm + sd2;
        } else {
            size = ovrthr;
        }
    }

    for (i32 j = 0; j < n; j++) {
        ss = SLC_DASUM(&n, &a[j * lda], &inc1);
        SLC_DSCAL(&n, &eavgev, &a[j * lda], &inc1);
        if (lbals) {
            bd = dwork[jworv1 + j];
            f64 cand = ss + sd2 / bd;
            if (size < cand) size = cand;
        }
    }

    f64 rerr = log10(big) + log10(eabs) - log10(small) -
               log10(emnorm) - log10(eps);
    f64 rerl;
    if (size > emnorm) {
        rerl = log10((size / emnorm - ONE) / eps);
    } else {
        rerl = ZERO;
    }

    i32 mdig_val = ndec - (i32)(rerr + HALF);
    if (mdig_val > ndecm1) mdig_val = ndecm1;
    i32 idig_val = ndec - (i32)(rerl + HALF);
    if (idig_val > ndecm1) idig_val = ndecm1;

    *mdig = mdig_val;
    *idig = idig_val;

    if (*mdig <= 0) {
        *mdig = 0;
        *iwarn = 1;
    }
    if (*idig <= 0) {
        *idig = 0;
        *iwarn = 2;
    }
}
