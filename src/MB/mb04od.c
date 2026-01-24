#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

void mb04od(const char* uplo_str, const i32 n, const i32 m, const i32 p,
            f64* r, const i32 ldr, f64* a, const i32 lda,
            f64* b, const i32 ldb, f64* c, const i32 ldc,
            f64* tau, f64* dwork)
{
    bool luplo;
    i32 i, im;
    char uplo = uplo_str[0];

    /* For efficiency reasons, the parameters are not checked (per SLICOT design) */
    if ((n < p ? n : p) == 0) {
        return;
    }

    luplo = (uplo == 'U' || uplo == 'u');

    if (luplo) {
        for (i = 0; i < n; i++) {
            im = (i < p) ? i : p;
            i32 imp1 = im + 1;
            i32 inc = 1;

            SLC_DLARFG(&imp1, &r[i + i*ldr], &a[i*lda], &inc, &tau[i]);

            if (n - i - 1 > 0) {
                i32 ncols = n - i - 1;
                SLC_MB04OY(im, ncols, &a[i*lda], tau[i], &r[i + (i+1)*ldr], ldr,
                           &a[(i+1)*lda], lda, dwork);
            }

            if (m > 0) {
                SLC_MB04OY(im, m, &a[i*lda], tau[i], &b[i], ldb, c, ldc, dwork);
            }
        }
    } else {
        for (i = 0; i < n - 1; i++) {
            i32 pp1 = p + 1;
            i32 inc = 1;

            SLC_DLARFG(&pp1, &r[i + i*ldr], &a[i*lda], &inc, &tau[i]);

            i32 ncols = n - i - 1;
            SLC_MB04OY(p, ncols, &a[i*lda], tau[i], &r[i + (i+1)*ldr], ldr,
                       &a[(i+1)*lda], lda, dwork);
        }

        {
            i32 pp1 = p + 1;
            i32 inc = 1;
            SLC_DLARFG(&pp1, &r[n-1 + (n-1)*ldr], &a[(n-1)*lda], &inc, &tau[n-1]);
        }

        if (m > 0) {
            for (i = 0; i < n; i++) {
                SLC_MB04OY(p, m, &a[i*lda], tau[i], &b[i], ldb, c, ldc, dwork);
            }
        }
    }
}
