// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

f64 ma02id(const char *typ, const char *norm, i32 n, const f64 *a, i32 lda,
           const f64 *qg, i32 ldqg, f64 *dwork) {

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

    if (norm_m && lsh) {
        value = SLC_DLANGE("M", &n, &n, a, &lda, dwork);
        if (n > 1) {
            for (i32 j = 0; j < n + 1; j++) {
                for (i32 i = 0; i < j - 1; i++) {
                    f64 temp = fabs(qg[i + j * ldqg]);
                    if (temp > value) value = temp;
                }
                for (i32 i = j + 1; i < n; i++) {
                    f64 temp = fabs(qg[i + j * ldqg]);
                    if (temp > value) value = temp;
                }
            }
        }
    } else if (norm_m) {
        i32 np1 = n + 1;
        f64 val1 = SLC_DLANGE("M", &n, &n, a, &lda, dwork);
        f64 val2 = SLC_DLANGE("M", &n, &np1, qg, &ldqg, dwork);
        value = (val1 > val2) ? val1 : val2;
    } else if ((norm_1 || norm_i) && lsh) {
        for (i32 i = 0; i < n; i++) {
            dwork[i] = ZERO;
        }

        for (i32 j = 0; j < n; j++) {
            f64 sum = ZERO;
            for (i32 i = 0; i < n; i++) {
                f64 temp = fabs(a[i + j * lda]);
                sum += temp;
                dwork[i] += temp;
            }
            dwork[n + j] = sum;
        }

        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < j - 1; i++) {
                f64 temp = fabs(qg[i + j * ldqg]);
                dwork[i] += temp;
                dwork[j - 1] += temp;
            }
            f64 sum = dwork[n + j];
            for (i32 i = j + 1; i < n; i++) {
                f64 temp = fabs(qg[i + j * ldqg]);
                sum += temp;
                dwork[n + i] += temp;
            }
            if (sum > value) value = sum;
        }
        for (i32 i = 0; i < n - 1; i++) {
            f64 temp = fabs(qg[i + n * ldqg]);
            dwork[i] += temp;
            dwork[n - 1] += temp;
        }
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
                f64 temp = fabs(a[i + j * lda]);
                sum += temp;
                dwork[i] += temp;
            }
            dwork[n + j] = sum;
        }

        f64 sum = dwork[n] + fabs(qg[0]);
        for (i32 i = 1; i < n; i++) {
            f64 temp = fabs(qg[i]);
            sum += temp;
            dwork[n + i] += temp;
        }
        if (sum > value) value = sum;

        for (i32 j = 1; j < n; j++) {
            for (i32 i = 0; i < j - 1; i++) {
                f64 temp = fabs(qg[i + j * ldqg]);
                dwork[i] += temp;
                dwork[j - 1] += temp;
            }
            dwork[j - 1] += fabs(qg[(j - 1) + j * ldqg]);
            sum = dwork[n + j] + fabs(qg[j + j * ldqg]);
            for (i32 i = j + 1; i < n; i++) {
                f64 temp = fabs(qg[i + j * ldqg]);
                sum += temp;
                dwork[n + i] += temp;
            }
            if (sum > value) value = sum;
        }
        for (i32 i = 0; i < n - 2; i++) {
            f64 temp = fabs(qg[i + n * ldqg]);
            dwork[i] += temp;
            dwork[n - 1] += temp;
        }
        dwork[n - 1] += fabs(qg[(n - 1) + n * ldqg]);
        for (i32 i = 0; i < n; i++) {
            if (dwork[i] > value) value = dwork[i];
        }
    } else if (norm_f && lsh) {
        f64 scale = ZERO;
        f64 sum = ONE;
        i32 int1 = 1;

        for (i32 j = 0; j < n; j++) {
            SLC_DLASSQ(&n, &a[j * lda], &int1, &scale, &sum);
        }

        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLASSQ(&nm1, &qg[1], &int1, &scale, &sum);
        }
        if (n > 2) {
            i32 nm2 = n - 2;
            SLC_DLASSQ(&nm2, &qg[2 + ldqg], &int1, &scale, &sum);
        }
        for (i32 j = 2; j < n - 1; j++) {
            i32 jm2 = j - 1;
            SLC_DLASSQ(&jm2, &qg[j * ldqg], &int1, &scale, &sum);
            i32 nmj = n - j - 1;
            SLC_DLASSQ(&nmj, &qg[(j + 1) + j * ldqg], &int1, &scale, &sum);
        }
        i32 nm2 = n - 2;
        SLC_DLASSQ(&nm2, &qg[(n - 1) * ldqg], &int1, &scale, &sum);
        i32 nm1 = n - 1;
        SLC_DLASSQ(&nm1, &qg[n * ldqg], &int1, &scale, &sum);
        value = sqrt(TWO) * scale * sqrt(sum);
    } else if (norm_f) {
        f64 scale = ZERO;
        f64 sum = ONE;
        f64 dscl = ZERO;
        f64 dsum = ONE;
        i32 int1 = 1;

        for (i32 j = 0; j < n; j++) {
            SLC_DLASSQ(&n, &a[j * lda], &int1, &scale, &sum);
        }

        SLC_DLASSQ(&int1, &qg[0], &int1, &dscl, &dsum);
        if (n > 1) {
            i32 nm1 = n - 1;
            SLC_DLASSQ(&nm1, &qg[1], &int1, &scale, &sum);
        }
        for (i32 j = 1; j < n; j++) {
            i32 jm2 = j - 1;
            if (jm2 > 0) {
                SLC_DLASSQ(&jm2, &qg[j * ldqg], &int1, &scale, &sum);
            }
            i32 two = 2;
            SLC_DLASSQ(&two, &qg[(j - 1) + j * ldqg], &int1, &dscl, &dsum);
            i32 nmj = n - j - 1;
            if (nmj > 0) {
                SLC_DLASSQ(&nmj, &qg[(j + 1) + j * ldqg], &int1, &scale, &sum);
            }
        }
        i32 nm1 = n - 1;
        SLC_DLASSQ(&nm1, &qg[n * ldqg], &int1, &scale, &sum);
        SLC_DLASSQ(&int1, &qg[(n - 1) + n * ldqg], &int1, &dscl, &dsum);
        f64 v1 = sqrt(TWO) * scale * sqrt(sum);
        f64 v2 = dscl * sqrt(dsum);
        value = SLC_DLAPY2(&v1, &v2);
    }

    return value;
}
