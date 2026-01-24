// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

f64 ma02mz(const char *norm, const char *uplo, i32 n, const c128 *a, i32 lda,
           f64 *dwork) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    f64 value = ZERO;

    if (n == 0) {
        return ZERO;
    }

    char norm_c = toupper((unsigned char)norm[0]);
    char uplo_c = toupper((unsigned char)uplo[0]);

    if (norm_c == 'M') {
        value = ZERO;
        if (uplo_c == 'U') {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < j; i++) {
                    f64 temp = cabs(a[i + j * lda]);
                    if (temp > value) value = temp;
                }
                f64 temp = fabs(cimag(a[j + j * lda]));
                if (temp > value) value = temp;
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                f64 temp = fabs(cimag(a[j + j * lda]));
                if (temp > value) value = temp;
                for (i32 i = j + 1; i < n; i++) {
                    temp = cabs(a[i + j * lda]);
                    if (temp > value) value = temp;
                }
            }
        }
    } else if (norm_c == 'I' || norm_c == 'O' || norm_c == '1') {
        value = ZERO;
        if (uplo_c == 'U') {
            for (i32 j = 0; j < n; j++) {
                f64 sum = ZERO;
                for (i32 i = 0; i < j; i++) {
                    f64 absa = cabs(a[i + j * lda]);
                    sum += absa;
                    dwork[i] += absa;
                }
                dwork[j] = sum + fabs(cimag(a[j + j * lda]));
            }
            for (i32 i = 0; i < n; i++) {
                if (dwork[i] > value) value = dwork[i];
            }
        } else {
            for (i32 i = 0; i < n; i++) {
                dwork[i] = ZERO;
            }
            for (i32 j = 0; j < n; j++) {
                f64 sum = dwork[j] + fabs(cimag(a[j + j * lda]));
                for (i32 i = j + 1; i < n; i++) {
                    f64 absa = cabs(a[i + j * lda]);
                    sum += absa;
                    dwork[i] += absa;
                }
                if (sum > value) value = sum;
            }
            if (dwork[n - 1] > value) value = dwork[n - 1];
        }
    } else if (norm_c == 'F' || norm_c == 'E') {
        f64 scale = ZERO;
        f64 sum = ONE;
        i32 int1 = 1;

        if (uplo_c == 'U') {
            for (i32 j = 1; j < n; j++) {
                i32 len = j;
                SLC_ZLASSQ(&len, &a[j * lda], &int1, &scale, &sum);
            }
        } else {
            for (i32 j = 0; j < n - 1; j++) {
                i32 len = n - j - 1;
                SLC_ZLASSQ(&len, &a[(j + 1) + j * lda], &int1, &scale, &sum);
            }
        }
        sum = TWO * sum;

        for (i32 i = 0; i < n; i++) {
            f64 absa = fabs(cimag(a[i + i * lda]));
            if (absa != ZERO) {
                if (scale < absa) {
                    sum = ONE + sum * (scale / absa) * (scale / absa);
                    scale = absa;
                } else {
                    sum = sum + (absa / scale) * (absa / scale);
                }
            }
        }
        value = scale * sqrt(sum);
    }

    return value;
}
