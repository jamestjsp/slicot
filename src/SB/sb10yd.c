/**
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>

void sb10yd(i32 discfl, i32 flag, i32 lendat, const f64* rfrdat, const f64* ifrdat,
            const f64* omega, i32* n, f64* a, i32 lda, f64* b, f64* c, f64* d,
            f64 tol, i32* iwork, f64* dwork, i32 ldwork, c128* zwork, i32 lzwork,
            i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 FOUR = 4.0;
    const f64 TEN = 10.0;
    const i32 HNPTS = 2048;
    i32 one = 1;

    /* Local variables */
    i32 i, ii, info2, ip1, ip2, istart, istop, iwa0, iwab, iwbmat, iwbp, iwbx,
        iwdme, iwdomo, iwmag, iws, iwvar, iwxi, iwxr, iwymag, k, lw1, lw2,
        lw3, lw4, mn, n1, n2, p, rank;
    f64 p1, p2, pi, pw, rat, tolb, toll;
    c128 *xhat = NULL;
    c128 zzero = 0.0 + 0.0 * I;
    c128 zone = 1.0 + 0.0 * I;

    *info = 0;
    pi = FOUR * atan(ONE);
    pw = omega[0];
    n1 = *n + 1;
    n2 = *n + n1;

    /* Test input parameters */
    if (discfl != 0 && discfl != 1) {
        *info = -1;
    } else if (flag != 0 && flag != 1) {
        *info = -2;
    } else if (lendat < 2) {
        *info = -3;
    } else if (pw < ZERO) {
        *info = -6;
    } else if (*n > lendat - 1) {
        *info = -7;
    } else if (lda < (*n > 1 ? *n : 1)) {
        *info = -9;
    } else {
        for (k = 1; k < lendat; k++) {
            if (omega[k] < pw) {
                *info = -6;
            }
            pw = omega[k];
        }
        if (discfl == 1 && omega[lendat - 1] > pi) {
            *info = -6;
        }
    }

    /* Workspace */
    i32 dlwmax = 0;
    i32 clwmax = 0;

    if (*info == 0) {
        lw1 = 2 * lendat + 4 * HNPTS;
        lw2 = lendat + 6 * HNPTS;
        mn = (2 * lendat < n2) ? 2 * lendat : n2;

        if (*n > 0) {
            i32 term1 = mn + 6 * *n + 4;
            i32 term2 = 2 * mn + 1;
            lw3 = 2 * lendat * n2 + (2 * lendat > n2 ? 2 * lendat : n2) +
                  (term1 > term2 ? term1 : term2);
        } else {
            lw3 = 4 * lendat + 5;
        }

        if (flag == 0) {
            lw4 = 0;
        } else {
            i32 term3 = *n * *n + 5 * *n;
            i32 term4 = 6 * *n + 1 + (1 < *n ? 1 : *n);
            lw4 = (term3 > term4) ? term3 : term4;
        }

        dlwmax = (2 > lw1) ? 2 : lw1;
        if (lw2 > dlwmax) dlwmax = lw2;
        if (lw3 > dlwmax) dlwmax = lw3;
        if (lw4 > dlwmax) dlwmax = lw4;

        if (*n > 0) {
            clwmax = lendat * (n2 + 2);
        } else {
            clwmax = lendat;
        }

        if (ldwork < dlwmax) {
            *info = -16;
        } else if (lzwork < clwmax) {
            *info = -18;
        }
    }

    if (*info != 0) {
        return;
    }

    /* Allocate xhat dynamically to avoid large stack allocation (HNPTS/2 * sizeof(c128) = 16KB) */
    xhat = (c128 *)malloc((HNPTS / 2) * sizeof(c128));
    if (xhat == NULL) {
        *info = -16;
        return;
    }

    /* Set tolerances */
    tolb = SLC_DLAMCH("Epsilon");
    toll = tol;
    if (toll <= ZERO) {
        toll = FOUR * (f64)(lendat * *n) * tolb;
    }

    /* Workspace usage 1 */
    iwdomo = 0;
    iwdme = iwdomo + lendat;
    iwymag = iwdme + 2 * HNPTS;
    iwmag = iwymag + 2 * HNPTS;

    /* Bilinear transformation */
    if (discfl == 0) {
        pw = sqrt(omega[0] * omega[lendat - 1] + sqrt(tolb));
        for (k = 0; k < lendat; k++) {
            dwork[iwdme + k] = (omega[k] / pw) * (omega[k] / pw);
            dwork[iwdomo + k] = acos((ONE - dwork[iwdme + k]) / (ONE + dwork[iwdme + k]));
        }
    } else {
        SLC_DCOPY(&lendat, omega, &one, &dwork[iwdomo], &one);
    }

    /* Linear interpolation */
    for (k = 0; k < lendat; k++) {
        dwork[iwmag + k] = SLC_DLAPY2(&rfrdat[k], &ifrdat[k]);
        dwork[iwmag + k] = (ONE / log(TEN)) * log(dwork[iwmag + k]);
    }

    for (k = 0; k < HNPTS; k++) {
        dwork[iwdme + k] = (f64)k * pi / (f64)HNPTS;
        dwork[iwymag + k] = ZERO;

        if (dwork[iwdme + k] < dwork[iwdomo]) {
            dwork[iwymag + k] = dwork[iwmag];
        } else if (dwork[iwdme + k] >= dwork[iwdomo + lendat - 1]) {
            dwork[iwymag + k] = dwork[iwmag + lendat - 1];
        }
    }

    for (i = 1; i < lendat; i++) {
        p1 = (f64)HNPTS * dwork[iwdomo + i - 1] / pi + ONE;
        ip1 = (i32)p1;
        if ((f64)ip1 != p1) ip1++;

        p2 = (f64)HNPTS * dwork[iwdomo + i] / pi + ONE;
        ip2 = (i32)p2;
        if ((f64)ip2 != p2) ip2++;

        for (p = ip1; p < ip2; p++) {
            i32 q = p - 1;
            if (q < 0 || q >= HNPTS) continue;

            rat = dwork[iwdme + q] - dwork[iwdomo + i - 1];
            rat = rat / (dwork[iwdomo + i] - dwork[iwdomo + i - 1]);
            dwork[iwymag + q] = (ONE - rat) * dwork[iwmag + i - 1] + rat * dwork[iwmag + i];
        }
    }

    for (k = 0; k < HNPTS; k++) {
        dwork[iwymag + k] = exp(log(TEN) * dwork[iwymag + k]);
    }

    /* Duplicate data around disc */
    for (k = 0; k < HNPTS; k++) {
        dwork[iwdme + HNPTS + k] = TWO * pi - dwork[iwdme + HNPTS - k - 1];
        dwork[iwymag + HNPTS + k] = dwork[iwymag + HNPTS - k - 1];
    }

    /* Complex cepstrum to get min phase: LOG (Magnitude) */
    for (k = 0; k < 2 * HNPTS; k++) {
        dwork[iwymag + k] = TWO * log(dwork[iwymag + k]);
    }

    /* Workspace usage 2 */
    iwxr = iwymag;
    iwxi = iwmag;

    for (k = 0; k < 2 * HNPTS; k++) {
        dwork[iwxi + k] = ZERO;
    }

    /* IFFT */
    dg01md("I", 2 * HNPTS, &dwork[iwxr], &dwork[iwxi], &info2);

    /* Rescale */
    f64 scale = ONE / (TWO * (f64)HNPTS);
    i32 two_hnpts = 2 * HNPTS;
    SLC_DSCAL(&two_hnpts, &scale, &dwork[iwxr], &one);
    SLC_DSCAL(&two_hnpts, &scale, &dwork[iwxi], &one);

    /* Halve the result at 0 */
    dwork[iwxr] /= TWO;
    dwork[iwxi] /= TWO;

    /* FFT */
    dg01md("D", HNPTS, &dwork[iwxr], &dwork[iwxi], &info2);

    /* Get the EXP of the result */
    for (k = 0; k < HNPTS / 2; k++) {
        f64 real_part = dwork[iwxr + k];
        f64 imag_part = dwork[iwxi + k];
        xhat[k] = exp(real_part) * (cos(imag_part) + I * sin(imag_part));
        dwork[iwdme + k] = dwork[iwdme + 2 * k];
    }

    /* Interpolate back to original frequency data */
    istart = 0;
    istop = lendat;

    for (i = 0; i < lendat; i++) {
        zwork[i] = zzero;
        if (dwork[iwdomo + i] <= dwork[iwdme]) {
            zwork[i] = xhat[0];
            istart = i + 1;
        } else if (dwork[iwdomo + i] >= dwork[iwdme + HNPTS / 2 - 1]) {
            zwork[i] = xhat[HNPTS / 2 - 1];
            istop--;
        }
    }

    for (i = istart; i < istop; i++) {
        ii = HNPTS / 2 - 1; /* 0-based index */
        p = ii;
        while (1) {
            if (dwork[iwdme + ii] >= dwork[iwdomo + i]) {
                p = ii;
            }
            ii--;
            if (ii < 0) break;
        }
        rat = (dwork[iwdomo + i] - dwork[iwdme + p - 1]) /
              (dwork[iwdme + p] - dwork[iwdme + p - 1]);
        zwork[i] = rat * xhat[p] + (ONE - rat) * xhat[p - 1];
    }

    /* CASE N > 0 */
    if (*n > 0) {
        /* Complex workspace usage 1 */
        iwa0 = lendat; /* zwork index offset */
        iwvar = iwa0 + lendat * n1;

        for (k = 0; k < lendat; k++) {
            if (discfl == 0) {
                zwork[iwvar + k] = cos(dwork[iwdomo + k]) + I * sin(dwork[iwdomo + k]);
            } else {
                zwork[iwvar + k] = cos(omega[k]) + I * sin(omega[k]);
            }
        }

        /* Array for DGELSY */
        for (k = 0; k < n2; k++) {
            iwork[k] = 0;
        }

        /* Constructing A0 */
        for (k = 0; k < lendat; k++) {
            zwork[iwa0 + *n * lendat + k] = zone;
        }

        for (i = 0; i < *n; i++) {
            /* Fortran loop 190 I=1,N. Reverse order for power? 
               ZWORK(IWA0+(N1-I)*LENDAT+K-1) is previous. 
               We are building powers of z/e^(jwt) basically. */
            /* C 0-based indexing:
               Fortran I=1 => (N-1)*LENDAT + K. Source (N1-1)*LENDAT -> N*LENDAT (the ones we just set)
             */
            /* Let's follow Fortran closely. I runs 1 to N.
               Target: (N-I)*LENDAT + K (0-based)
               Source: (N1-I)*LENDAT + K (0-based) = (N+1-I)*LENDAT+K
               Wait, Loop I=1, Target idx = (N-1)*LENDAT.. Source idx = N*LENDAT.. 
               So we are going down from N-1. */
             
            for (k = 0; k < lendat; k++) {
                 zwork[iwa0 + (*n - 1 - i) * lendat + k] =
                    zwork[iwa0 + (*n - i) * lendat + k] * zwork[iwvar + k];
            }
        }

        /* Complex workspace usage 2 */
        iwbp = iwvar;
        iwab = iwbp + lendat;

        /* Constructing BP */
        for (k = 0; k < lendat; k++) {
            zwork[iwbp + k] = zwork[iwa0 + k] * zwork[k];
        }

        /* Constructing AB */
        for (i = 0; i < *n; i++) {
            for (k = 0; k < lendat; k++) {
                zwork[iwab + i * lendat + k] = -zwork[k] * zwork[iwa0 + (i + 1) * lendat + k];
            }
        }

        /* Workspace usage 3 */
        iwbx = 2 * lendat * n2; /* dwork index */
        iws = iwbx + (2 * lendat > n2 ? 2 * lendat : n2);

        /* Constructing AX - split complex A0 and AB into real parts in DWORK */
        for (i = 0; i < n1; i++) {
             for (k = 0; k < lendat; k++) {
                 dwork[2 * i * lendat + k] = creal(zwork[iwa0 + i * lendat + k]);
                 dwork[(2 * i + 1) * lendat + k] = cimag(zwork[iwa0 + i * lendat + k]);
             }
        }
        for (i = 0; i < *n; i++) {
            for (k = 0; k < lendat; k++) {
                dwork[2 * n1 * lendat + 2 * i * lendat + k] = creal(zwork[iwab + i * lendat + k]);
                dwork[2 * n1 * lendat + (2 * i + 1) * lendat + k] = cimag(zwork[iwab + i * lendat + k]);
            }
        }

        /* Constructing BX */
        for (k = 0; k < lendat; k++) {
            dwork[iwbx + k] = creal(zwork[iwbp + k]);
            dwork[iwbx + lendat + k] = cimag(zwork[iwbp + k]);
        }

        /* Estimating X using DGELSY */
        i32 m_ls = 2 * lendat;
        i32 n_ls = n2;
        i32 nrhs = 1;
        i32 ld_ls = 2 * lendat;
        i32 ld_rhs = (2 * lendat > n2 ? 2 * lendat : n2);
        i32 ldwork_ls = ldwork - iws;

        SLC_DGELSY(&m_ls, &n_ls, &nrhs, dwork, &ld_ls, &dwork[iwbx], &ld_rhs, iwork, &toll, &rank,
                   &dwork[iws], &ldwork_ls, &info2);
        
        if ((i32)dwork[iws] + iws - 1 > dlwmax) dlwmax = (i32)dwork[iws] + iws - 1;

        /* Constructing A matrix */
        for (k = 0; k < *n; k++) {
            a[k] = -dwork[iwbx + n1 + k];
        }
        if (*n > 1) {
            i32 n_1 = *n - 1;
            SLC_DLASET("Full", n, &n_1, &ZERO, &ONE, &a[lda], &lda);
        }

        /* Constructing B matrix */
        for (k = 0; k < *n; k++) {
            b[k] = dwork[iwbx + n1 + k] * dwork[iwbx] - dwork[iwbx + k + 1];
        }

        /* Constructing C matrix */
        c[0] = -ONE;
        for (k = 1; k < *n; k++) {
            c[k] = ZERO;
        }

        /* Constructing D matrix */
        d[0] = dwork[iwbx];

        /* Transform to continuous-time case if needed */
        if (discfl == 0) {
            ab04md('D', *n, 1, 1, ONE, pw, a, lda, b, *n, c, 1, d, 1, iwork, dwork, ldwork);
        }

        /* Enforce stability/min phase */
        if (flag == 1) {
            sb10zp(discfl, n, a, lda, b, c, d, iwork, dwork, ldwork, info);
            if (*info != 0) {
                free(xhat);
                return;
            }
        }

    } else {
        /* CASE N = 0 */
        iwbmat = 2 * lendat; /* dwork index */
        iws = iwbmat + 2 * lendat;

        for (k = 0; k < lendat; k++) {
            dwork[k] = ONE;
            dwork[k + lendat] = ZERO;
            dwork[iwbmat + k] = creal(zwork[k]);
            dwork[iwbmat + lendat + k] = cimag(zwork[k]);
        }

        iwork[0] = 0;
        i32 m_ls = 2 * lendat;
        i32 n_ls = 1;
        i32 nrhs = 1;
        i32 ld_ls = 2 * lendat;
        i32 ld_rhs = 2 * lendat; 
        i32 ldwork_ls = ldwork - iws;

        SLC_DGELSY(&m_ls, &n_ls, &nrhs, dwork, &ld_ls, &dwork[iwbmat], &ld_rhs, iwork, &toll, &rank,
                   &dwork[iws], &ldwork_ls, &info2);

        d[0] = dwork[iwbmat];
    }

    dwork[0] = (f64)dlwmax;
    /* dwork[1] = clwmax; SB10YD Fortran puts complex workspace requirement in DWORK(2).
       But DWORK is double. So C wrapper might handle this differently.
       Fortran: DWORK(2) = CLWMAX.
       We'll set dwork[1] = (f64)clwmax. */
    dwork[1] = (f64)clwmax;
    free(xhat);
}
