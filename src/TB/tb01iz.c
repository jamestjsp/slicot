/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

static f64 cabs1(c128 z) {
    return fabs(creal(z)) + fabs(cimag(z));
}

i32 slicot_tb01iz(char job, i32 n, i32 m, i32 p, f64* maxred,
                  c128* a, i32 lda, c128* b, i32 ldb, c128* c, i32 ldc,
                  f64* scale) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 SCLFAC = 10.0;
    const f64 FACTOR = 0.95;
    const f64 MAXR = 10.0;

    i32 info = 0;
    char job_up = (char)toupper((unsigned char)job);
    bool withb = (job_up == 'A') || (job_up == 'B');
    bool withc = (job_up == 'A') || (job_up == 'C');

    if (!withb && !withc && job_up != 'N') {
        info = -1;
    } else if (n < 0) {
        info = -2;
    } else if (m < 0) {
        info = -3;
    } else if (p < 0) {
        info = -4;
    } else if (*maxred > ZERO && *maxred < ONE) {
        info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        info = -7;
    } else if ((m > 0 && ldb < (n > 1 ? n : 1)) || (m == 0 && ldb < 1)) {
        info = -9;
    } else if (ldc < (p > 1 ? p : 1)) {
        info = -11;
    }

    if (info != 0) {
        i32 xinfo = -info;
        SLC_XERBLA("TB01IZ", &xinfo);
        return info;
    }

    if (n == 0) {
        return 0;
    }

    f64 snorm = ZERO;
    i32 one = 1;

    for (i32 j = 0; j < n; j++) {
        scale[j] = ONE;
        f64 co = SLC_DZASUM(&n, &a[j * lda], &one);
        if (withc && p > 0) {
            co += SLC_DZASUM(&p, &c[j * ldc], &one);
        }
        if (co > snorm) snorm = co;
    }

    if (withb) {
        for (i32 j = 0; j < m; j++) {
            f64 col_norm = SLC_DZASUM(&n, &b[j * ldb], &one);
            if (col_norm > snorm) snorm = col_norm;
        }
    }

    if (snorm == ZERO) {
        return 0;
    }

    f64 sfmin1 = SLC_DLAMCH("S") / SLC_DLAMCH("P");
    f64 sfmax1 = ONE / sfmin1;
    f64 sfmin2 = sfmin1 * SCLFAC;
    f64 sfmax2 = ONE / sfmin2;

    f64 sred = *maxred;
    if (sred <= ZERO) sred = MAXR;

    f64 maxnrm = snorm / sred;
    if (maxnrm < sfmin1) maxnrm = sfmin1;

    bool noconv;
    do {
        noconv = false;

        for (i32 i = 0; i < n; i++) {
            f64 co = ZERO;
            f64 ro = ZERO;

            for (i32 j = 0; j < n; j++) {
                if (j == i) continue;
                co += cabs1(a[j + i * lda]);
                ro += cabs1(a[i + j * lda]);
            }

            i32 ica = SLC_IZAMAX(&n, &a[i * lda], &one) - 1;
            f64 ca = cabs(a[ica + i * lda]);
            i32 ira = SLC_IZAMAX(&n, &a[i], &lda) - 1;
            f64 ra = cabs(a[i + ira * lda]);

            if (withc && p > 0) {
                co += SLC_DZASUM(&p, &c[i * ldc], &one);
                i32 ica_c = SLC_IZAMAX(&p, &c[i * ldc], &one) - 1;
                f64 ca_c = cabs(c[ica_c + i * ldc]);
                if (ca_c > ca) ca = ca_c;
            }

            if (withb && m > 0) {
                ro += SLC_DZASUM(&m, &b[i], &ldb);
                i32 ira_b = SLC_IZAMAX(&m, &b[i], &ldb) - 1;
                f64 ra_b = cabs(b[i + ira_b * ldb]);
                if (ra_b > ra) ra = ra_b;
            }

            if (co == ZERO && ro == ZERO) continue;

            if (co == ZERO) {
                if (ro <= maxnrm) continue;
                co = maxnrm;
            }
            if (ro == ZERO) {
                if (co <= maxnrm) continue;
                ro = maxnrm;
            }

            f64 g = ro / SCLFAC;
            f64 f = ONE;
            f64 s = co + ro;

            while (co < g &&
                   fmax(f, fmax(co, ca)) < sfmax2 &&
                   fmin(ro, fmin(g, ra)) > sfmin2) {
                f *= SCLFAC;
                co *= SCLFAC;
                ca *= SCLFAC;
                g /= SCLFAC;
                ro /= SCLFAC;
                ra /= SCLFAC;
            }

            g = co / SCLFAC;

            while (g >= ro &&
                   fmax(ro, ra) < sfmax2 &&
                   fmin(f, fmin(co, fmin(g, ca))) > sfmin2) {
                f /= SCLFAC;
                co /= SCLFAC;
                ca /= SCLFAC;
                g /= SCLFAC;
                ro *= SCLFAC;
                ra *= SCLFAC;
            }

            if ((co + ro) >= FACTOR * s) continue;

            if (f < ONE && scale[i] < ONE) {
                if (f * scale[i] <= sfmin1) continue;
            }
            if (f > ONE && scale[i] > ONE) {
                if (scale[i] >= sfmax1 / f) continue;
            }

            g = ONE / f;
            scale[i] *= f;
            noconv = true;

            SLC_ZDSCAL(&n, &g, &a[i], &lda);
            SLC_ZDSCAL(&n, &f, &a[i * lda], &one);
            if (m > 0) SLC_ZDSCAL(&m, &g, &b[i], &ldb);
            if (p > 0) SLC_ZDSCAL(&p, &f, &c[i * ldc], &one);
        }
    } while (noconv);

    *maxred = snorm;
    snorm = ZERO;

    for (i32 j = 0; j < n; j++) {
        f64 co = SLC_DZASUM(&n, &a[j * lda], &one);
        if (withc && p > 0) {
            co += SLC_DZASUM(&p, &c[j * ldc], &one);
        }
        if (co > snorm) snorm = co;
    }

    if (withb) {
        for (i32 j = 0; j < m; j++) {
            f64 col_norm = SLC_DZASUM(&n, &b[j * ldb], &one);
            if (col_norm > snorm) snorm = col_norm;
        }
    }

    *maxred = *maxred / snorm;

    return info;
}
