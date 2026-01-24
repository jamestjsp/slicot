#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void mb04iy(
    const char* side,
    const char* trans,
    const i32 n,
    const i32 m,
    const i32 k,
    const i32 p,
    f64* a,
    const i32 lda,
    const f64* tau,
    f64* c,
    const i32 ldc,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 one = 1.0;
    bool left, tran;
    i32 i, minval, i_lapack;
    f64 aii, wrkopt;

    *info = 0;
    left = (side[0] == 'L' || side[0] == 'l');
    tran = (trans[0] == 'T' || trans[0] == 't');

    // Parameter validation
    if (!left && !(side[0] == 'R' || side[0] == 'r')) {
        *info = -1;
    } else if (!tran && !(trans[0] == 'N' || trans[0] == 'n')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (k < 0 || (left && k > n) || (!left && k > m)) {
        *info = -5;
    } else if (p < 0) {
        *info = -6;
    } else if ((left && lda < (n > 0 ? n : 1)) ||
               (!left && lda < (m > 0 ? m : 1))) {
        *info = -8;
    } else if (ldc < (n > 0 ? n : 1)) {
        *info = -11;
    } else if ((left && ldwork < (m > 0 ? m : 1)) ||
               (!left && ldwork < (n > 0 ? n : 1))) {
        *info = -13;
    }

    if (*info != 0) {
        return;
    }

    // Quick return
    if (m == 0 || n == 0 || k == 0 ||
        (left && n < p) || (!left && m < p)) {
        dwork[0] = one;
        return;
    }

    if (left) {
        wrkopt = (f64)m;
        if (tran) {
            // Apply H(i) to C(i:i+n-p-1,1:m) for i=1:min(k,p)
            minval = (k < p) ? k : p;
            for (i = 0; i < minval; i++) {
                aii = a[i + i * lda];
                a[i + i * lda] = one;
                SLC_DLARF(side, &(i32){n - p}, &m,
                         &a[i + i * lda], &(i32){1},
                         &tau[i],
                         &c[i + 0 * ldc], &ldc,
                         dwork);
                a[i + i * lda] = aii;
            }

            if (p < n && p < k) {
                // Apply H(i) for i=p+1:k using DORMQR
                i32 nrows = n - p;
                i32 nrefl = k - p;
                SLC_DORMQR(side, trans,
                          &nrows, &m, &nrefl,
                          &a[p + p * lda], &lda,
                          &tau[p],
                          &c[p + 0 * ldc], &ldc,
                          dwork, &ldwork, &i_lapack);
                if (dwork[0] > wrkopt) {
                    wrkopt = dwork[0];
                }
            }

        } else {
            // TRANS='N': Apply in reverse order

            if (p < n && p < k) {
                // Apply H(i) for i=k:p+1:-1 using DORMQR
                i32 nrows = n - p;
                i32 nrefl = k - p;
                SLC_DORMQR(side, trans,
                          &nrows, &m, &nrefl,
                          &a[p + p * lda], &lda,
                          &tau[p],
                          &c[p + 0 * ldc], &ldc,
                          dwork, &ldwork, &i_lapack);
                if (dwork[0] > wrkopt) {
                    wrkopt = dwork[0];
                }
            }

            // Apply H(i) for i=min(k,p):1:-1
            minval = (k < p) ? k : p;
            for (i = minval - 1; i >= 0; i--) {
                aii = a[i + i * lda];
                a[i + i * lda] = one;
                SLC_DLARF(side, &(i32){n - p}, &m,
                         &a[i + i * lda], &(i32){1},
                         &tau[i],
                         &c[i + 0 * ldc], &ldc,
                         dwork);
                a[i + i * lda] = aii;
            }
        }

    } else {
        // SIDE='R'
        wrkopt = (f64)n;
        if (tran) {
            // TRANS='T': Apply in reverse order from right

            if (p < m && p < k) {
                // Apply H(i) for i=k:p+1:-1 using DORMQR
                i32 ncols = m - p;
                i32 nrefl = k - p;
                SLC_DORMQR(side, trans,
                          &n, &ncols, &nrefl,
                          &a[p + p * lda], &lda,
                          &tau[p],
                          &c[0 + p * ldc], &ldc,
                          dwork, &ldwork, &i_lapack);
                if (dwork[0] > wrkopt) {
                    wrkopt = dwork[0];
                }
            }

            // Apply H(i) for i=min(k,p):1:-1
            minval = (k < p) ? k : p;
            for (i = minval - 1; i >= 0; i--) {
                aii = a[i + i * lda];
                a[i + i * lda] = one;
                SLC_DLARF(side, &n, &(i32){m - p},
                         &a[i + i * lda], &(i32){1},
                         &tau[i],
                         &c[0 + i * ldc], &ldc,
                         dwork);
                a[i + i * lda] = aii;
            }

        } else {
            // TRANS='N': Forward order from right

            // Apply H(i) for i=1:min(k,p)
            minval = (k < p) ? k : p;
            for (i = 0; i < minval; i++) {
                aii = a[i + i * lda];
                a[i + i * lda] = one;
                SLC_DLARF(side, &n, &(i32){m - p},
                         &a[i + i * lda], &(i32){1},
                         &tau[i],
                         &c[0 + i * ldc], &ldc,
                         dwork);
                a[i + i * lda] = aii;
            }

            if (p < m && p < k) {
                // Apply H(i) for i=p+1:k using DORMQR
                i32 ncols = m - p;
                i32 nrefl = k - p;
                SLC_DORMQR(side, trans,
                          &n, &ncols, &nrefl,
                          &a[p + p * lda], &lda,
                          &tau[p],
                          &c[0 + p * ldc], &ldc,
                          dwork, &ldwork, &i_lapack);
                if (dwork[0] > wrkopt) {
                    wrkopt = dwork[0];
                }
            }
        }
    }

    dwork[0] = wrkopt;
}
