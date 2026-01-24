// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb01ud(const char* side, const char* trans, i32 m, i32 n,
            f64 alpha, f64* h, i32 ldh, const f64* a, i32 lda,
            f64* b, i32 ldb, i32* info)
{
    const f64 zero = 0.0;
    i32 int1 = 1;

    char side_c = (char)toupper((unsigned char)side[0]);
    char trans_c = (char)toupper((unsigned char)trans[0]);

    bool lside = (side_c == 'L');
    bool ltrans = (trans_c == 'T' || trans_c == 'C');

    *info = 0;

    if (!lside && side_c != 'R') {
        *info = -1;
    } else if (!ltrans && trans_c != 'N') {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldh < 1 || (lside && ldh < m) || (!lside && ldh < n)) {
        *info = -7;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -9;
    } else if (ldb < (m > 1 ? m : 1)) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    if (m == 0 || n == 0) {
        return;
    }

    if (alpha == zero) {
        SLC_DLASET("Full", &m, &n, &zero, &zero, b, &ldb);
        return;
    }

    SLC_DLACPY("Full", &m, &n, a, &lda, b, &ldb);
    SLC_DTRMM(&side_c, "Upper", &trans_c, "Non-unit", &m, &n, &alpha, h, &ldh, b, &ldb);

    if (lside) {
        if (m > 2) {
            i32 len = m - 2;
            i32 ldh1 = ldh + 1;
            SLC_DSWAP(&len, &h[2 + ldh], &ldh1, &h[2], &int1);
        }

        if (ltrans) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < m - 1; i++) {
                    b[i + j * ldb] += alpha * h[i + 1] * a[i + 1 + j * lda];
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 1; i < m; i++) {
                    b[i + j * ldb] += alpha * h[i] * a[i - 1 + j * lda];
                }
            }
        }

        if (m > 2) {
            i32 len = m - 2;
            i32 ldh1 = ldh + 1;
            SLC_DSWAP(&len, &h[2 + ldh], &ldh1, &h[2], &int1);
        }
    } else {
        if (ltrans) {
            for (i32 j = 0; j < n - 1; j++) {
                f64 hval = h[j + 1 + j * ldh];
                if (hval != zero) {
                    f64 scale = alpha * hval;
                    SLC_DAXPY(&m, &scale, &a[j * lda], &int1, &b[(j + 1) * ldb], &int1);
                }
            }
        } else {
            for (i32 j = 0; j < n - 1; j++) {
                f64 hval = h[j + 1 + j * ldh];
                if (hval != zero) {
                    f64 scale = alpha * hval;
                    SLC_DAXPY(&m, &scale, &a[(j + 1) * lda], &int1, &b[j * ldb], &int1);
                }
            }
        }
    }
}
