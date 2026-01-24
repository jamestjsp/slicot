/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <complex.h>

f64 ma02iz(const char *typ, const char *norm, i32 n, const c128 *a, i32 lda,
           const c128 *qg, i32 ldqg, f64 *dwork) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    bool lsh = (typ[0] == 'S' || typ[0] == 's');
    bool norm_m = (norm[0] == 'M' || norm[0] == 'm');
    bool norm_1 = (norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1');
    bool norm_i = (norm[0] == 'I' || norm[0] == 'i');
    bool norm_f = (norm[0] == 'F' || norm[0] == 'f' || norm[0] == 'E' || norm[0] == 'e');

    f64 value = ZERO;

    if (n == 0) {
        return ZERO;
    }

    f64 dum[2];

    if (norm_m && lsh) {
        value = SLC_ZLANGE("M", &n, &n, a, &lda, dum);
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < j - 1; i++) {
                f64 temp = cabs(qg[i + j * ldqg]);
                if (temp > value) value = temp;
            }
            f64 temp = fabs(cimag(qg[j + j * ldqg]));
            if (temp > value) value = temp;
            for (i32 i = j + 1; i < n; i++) {
                temp = cabs(qg[i + j * ldqg]);
                if (temp > value) value = temp;
            }
            temp = fabs(cimag(qg[j + (j + 1) * ldqg]));
            if (temp > value) value = temp;
        }
        for (i32 i = 0; i < n - 1; i++) {
            f64 temp = cabs(qg[i + n * ldqg]);
            if (temp > value) value = temp;
        }
    } else if (norm_m) {
        value = SLC_ZLANGE("M", &n, &n, a, &lda, dum);
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < j - 1; i++) {
                f64 temp = cabs(qg[i + j * ldqg]);
                if (temp > value) value = temp;
            }
            f64 temp = fabs(creal(qg[j + j * ldqg]));
            if (temp > value) value = temp;
            for (i32 i = j + 1; i < n; i++) {
                temp = cabs(qg[i + j * ldqg]);
                if (temp > value) value = temp;
            }
            temp = fabs(creal(qg[j + (j + 1) * ldqg]));
            if (temp > value) value = temp;
        }
        for (i32 i = 0; i < n - 1; i++) {
            f64 temp = cabs(qg[i + n * ldqg]);
            if (temp > value) value = temp;
        }
    } else if ((norm_1 || norm_i) && lsh) {
        for (i32 i = 0; i < n; i++) {
            dwork[i] = ZERO;
        }

        for (i32 j = 0; j < n; j++) {
            f64 sum = ZERO;
            for (i32 i = 0; i < n; i++) {
                f64 temp = cabs(a[i + j * lda]);
                sum += temp;
                dwork[i] += temp;
            }
            dwork[n + j] = sum;
        }

        f64 sum = dwork[n] + fabs(cimag(qg[0]));
        for (i32 i = 1; i < n; i++) {
            f64 temp = cabs(qg[i]);
            sum += temp;
            dwork[n + i] += temp;
        }
        if (sum > value) value = sum;

        for (i32 j = 1; j < n; j++) {
            for (i32 i = 0; i < j - 1; i++) {
                f64 temp = cabs(qg[i + j * ldqg]);
                dwork[i] += temp;
                dwork[j - 1] += temp;
            }
            dwork[j - 1] += fabs(cimag(qg[(j - 1) + j * ldqg]));
            sum = dwork[n + j] + fabs(cimag(qg[j + j * ldqg]));
            for (i32 i = j + 1; i < n; i++) {
                f64 temp = cabs(qg[i + j * ldqg]);
                sum += temp;
                dwork[n + i] += temp;
            }
            if (sum > value) value = sum;
        }
        for (i32 i = 0; i < n - 1; i++) {
            f64 temp = cabs(qg[i + n * ldqg]);
            dwork[i] += temp;
            dwork[n - 1] += temp;
        }
        dwork[n - 1] += fabs(cimag(qg[(n - 1) + n * ldqg]));
        for (i32 i = 0; i < n; i++) {
            if (dwork[i] > value) value = dwork[i];
        }
    } else if (norm_1 || norm_i) {
        for (i32 i = 0; i < n; i++) {
            dwork[i] = ZERO;
        }

        for (i32 j = 0; j < n; j++) {
            f64 sum = ZERO;
            for (i32 i = 0; i < n; i++) {
                f64 temp = cabs(a[i + j * lda]);
                sum += temp;
                dwork[i] += temp;
            }
            dwork[n + j] = sum;
        }

        f64 sum = dwork[n] + fabs(creal(qg[0]));
        for (i32 i = 1; i < n; i++) {
            f64 temp = cabs(qg[i]);
            sum += temp;
            dwork[n + i] += temp;
        }
        if (sum > value) value = sum;

        for (i32 j = 1; j < n; j++) {
            for (i32 i = 0; i < j - 1; i++) {
                f64 temp = cabs(qg[i + j * ldqg]);
                dwork[i] += temp;
                dwork[j - 1] += temp;
            }
            dwork[j - 1] += fabs(creal(qg[(j - 1) + j * ldqg]));
            sum = dwork[n + j] + fabs(creal(qg[j + j * ldqg]));
            for (i32 i = j + 1; i < n; i++) {
                f64 temp = cabs(qg[i + j * ldqg]);
                sum += temp;
                dwork[n + i] += temp;
            }
            if (sum > value) value = sum;
        }
        for (i32 i = 0; i < n - 1; i++) {
            f64 temp = cabs(qg[i + n * ldqg]);
            dwork[i] += temp;
            dwork[n - 1] += temp;
        }
        dwork[n - 1] += fabs(creal(qg[(n - 1) + n * ldqg]));
        for (i32 i = 0; i < n; i++) {
            if (dwork[i] > value) value = dwork[i];
        }
    } else if (norm_f && lsh) {
        f64 scale = ZERO;
        f64 sum = ONE;
        i32 int1 = 1;

        for (i32 j = 0; j < n; j++) {
            SLC_ZLASSQ(&n, &a[j * lda], &int1, &scale, &sum);
        }

        f64 dscl = fabs(cimag(qg[0]));
        f64 dsum = ONE;

        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_ZLASSQ(&nm1, &qg[1], &int1, &scale, &sum);
            dum[0] = cimag(qg[ldqg]);
            dum[1] = cimag(qg[1 + ldqg]);
            i32 two = 2;
            SLC_DLASSQ(&two, dum, &int1, &dscl, &dsum);
        }
        if (n > 2) {
            i32 nm2 = n - 2;
            SLC_ZLASSQ(&nm2, &qg[2 + ldqg], &int1, &scale, &sum);
        }
        for (i32 j = 2; j < n; j++) {
            i32 jm2 = j - 1;
            SLC_ZLASSQ(&jm2, &qg[j * ldqg], &int1, &scale, &sum);
            dum[0] = cimag(qg[(j - 1) + j * ldqg]);
            dum[1] = cimag(qg[j + j * ldqg]);
            i32 two = 2;
            SLC_DLASSQ(&two, dum, &int1, &dscl, &dsum);
            i32 nmj = n - j - 1;
            if (nmj > 0) {
                SLC_ZLASSQ(&nmj, &qg[(j + 1) + j * ldqg], &int1, &scale, &sum);
            }
        }
        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_ZLASSQ(&nm1, &qg[n * ldqg], &int1, &scale, &sum);
        }
        dum[0] = cimag(qg[(n - 1) + n * ldqg]);
        SLC_DLASSQ(&int1, dum, &int1, &dscl, &dsum);

        f64 v1 = sqrt(TWO) * scale * sqrt(sum);
        f64 v2 = dscl * sqrt(dsum);
        value = SLC_DLAPY2(&v1, &v2);
    } else if (norm_f) {
        f64 scale = ZERO;
        f64 sum = ONE;
        i32 int1 = 1;

        for (i32 j = 0; j < n; j++) {
            SLC_ZLASSQ(&n, &a[j * lda], &int1, &scale, &sum);
        }

        f64 dscl = fabs(creal(qg[0]));
        f64 dsum = ONE;

        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_ZLASSQ(&nm1, &qg[1], &int1, &scale, &sum);
            dum[0] = creal(qg[ldqg]);
            dum[1] = creal(qg[1 + ldqg]);
            i32 two = 2;
            SLC_DLASSQ(&two, dum, &int1, &dscl, &dsum);
        }
        if (n > 2) {
            i32 nm2 = n - 2;
            SLC_ZLASSQ(&nm2, &qg[2 + ldqg], &int1, &scale, &sum);
        }
        for (i32 j = 2; j < n; j++) {
            i32 jm2 = j - 1;
            SLC_ZLASSQ(&jm2, &qg[j * ldqg], &int1, &scale, &sum);
            dum[0] = creal(qg[(j - 1) + j * ldqg]);
            dum[1] = creal(qg[j + j * ldqg]);
            i32 two = 2;
            SLC_DLASSQ(&two, dum, &int1, &dscl, &dsum);
            i32 nmj = n - j - 1;
            if (nmj > 0) {
                SLC_ZLASSQ(&nmj, &qg[(j + 1) + j * ldqg], &int1, &scale, &sum);
            }
        }
        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_ZLASSQ(&nm1, &qg[n * ldqg], &int1, &scale, &sum);
        }
        dum[0] = creal(qg[(n - 1) + n * ldqg]);
        SLC_DLASSQ(&int1, dum, &int1, &dscl, &dsum);

        f64 v1 = sqrt(TWO) * scale * sqrt(sum);
        f64 v2 = dscl * sqrt(dsum);
        value = SLC_DLAPY2(&v1, &v2);
    }

    return value;
}
