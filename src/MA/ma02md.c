// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

f64 ma02md(const char *norm, const char *uplo, i32 n, const f64 *a, i32 lda,
           f64 *dwork) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    f64 value = ZERO;

    if (n <= 1) {
        return ZERO;
    }

    char norm_c = toupper((unsigned char)norm[0]);
    char uplo_c = toupper((unsigned char)uplo[0]);

    if (norm_c == 'M') {
        value = ZERO;
        if (uplo_c == 'U') {
            for (i32 j = 1; j < n; j++) {
                for (i32 i = 0; i < j; i++) {
                    f64 temp = fabs(a[i + j * lda]);
                    if (temp > value) value = temp;
                }
            }
        } else {
            for (i32 j = 0; j < n - 1; j++) {
                for (i32 i = j + 1; i < n; i++) {
                    f64 temp = fabs(a[i + j * lda]);
                    if (temp > value) value = temp;
                }
            }
        }
    } else if (norm_c == 'I' || norm_c == 'O' || norm_c == '1') {
        value = ZERO;
        if (uplo_c == 'U') {
            dwork[0] = ZERO;
            for (i32 j = 1; j < n; j++) {
                f64 sum = ZERO;
                for (i32 i = 0; i < j; i++) {
                    f64 absa = fabs(a[i + j * lda]);
                    sum += absa;
                    dwork[i] += absa;
                }
                dwork[j] = sum;
            }
            for (i32 i = 0; i < n; i++) {
                if (dwork[i] > value) value = dwork[i];
            }
        } else {
            for (i32 i = 0; i < n; i++) {
                dwork[i] = ZERO;
            }
            for (i32 j = 0; j < n - 1; j++) {
                f64 sum = dwork[j];
                for (i32 i = j + 1; i < n; i++) {
                    f64 absa = fabs(a[i + j * lda]);
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
                SLC_DLASSQ(&len, &a[j * lda], &int1, &scale, &sum);
            }
        } else {
            for (i32 j = 0; j < n - 1; j++) {
                i32 len = n - j - 1;
                SLC_DLASSQ(&len, &a[(j + 1) + j * lda], &int1, &scale, &sum);
            }
        }
        value = scale * sqrt(TWO * sum);
    }

    return value;
}
