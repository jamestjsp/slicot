// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"

void SLC_MB04NY(i32 m, i32 n, const f64* v, i32 incv, f64 tau,
                f64* a, i32 lda, f64* b, i32 ldb, f64* dwork) {
    const f64 zero = 0.0;
    const f64 one = 1.0;

    if (tau == zero) {
        return;
    }

    i32 order = n + 1;

    if (order == 1) {
        f64 t1 = one - tau;
        for (i32 j = 0; j < m; j++) {
            a[j] *= t1;
        }
        return;
    }

    i32 iv = 0;
    if (incv < 0) {
        iv = (-n + 1) * incv;
    }

    if (order == 2) {
        f64 v1 = v[iv];
        f64 t1 = tau * v1;
        for (i32 j = 0; j < m; j++) {
            f64 sum = a[j] + v1 * b[j];
            a[j] -= sum * tau;
            b[j] -= sum * t1;
        }
        return;
    }

    if (order == 3) {
        f64 v1 = v[iv];
        f64 t1 = tau * v1;
        iv += incv;
        f64 v2 = v[iv];
        f64 t2 = tau * v2;
        for (i32 j = 0; j < m; j++) {
            f64 sum = a[j] + v1 * b[j] + v2 * b[j + ldb];
            a[j] -= sum * tau;
            b[j] -= sum * t1;
            b[j + ldb] -= sum * t2;
        }
        return;
    }

    if (order == 4) {
        f64 v1 = v[iv];
        f64 t1 = tau * v1;
        iv += incv;
        f64 v2 = v[iv];
        f64 t2 = tau * v2;
        iv += incv;
        f64 v3 = v[iv];
        f64 t3 = tau * v3;
        for (i32 j = 0; j < m; j++) {
            f64 sum = a[j] + v1 * b[j] + v2 * b[j + ldb] + v3 * b[j + 2 * ldb];
            a[j] -= sum * tau;
            b[j] -= sum * t1;
            b[j + ldb] -= sum * t2;
            b[j + 2 * ldb] -= sum * t3;
        }
        return;
    }

    if (order == 5) {
        f64 v1 = v[iv];
        f64 t1 = tau * v1;
        iv += incv;
        f64 v2 = v[iv];
        f64 t2 = tau * v2;
        iv += incv;
        f64 v3 = v[iv];
        f64 t3 = tau * v3;
        iv += incv;
        f64 v4 = v[iv];
        f64 t4 = tau * v4;
        for (i32 j = 0; j < m; j++) {
            f64 sum = a[j] + v1 * b[j] + v2 * b[j + ldb] +
                      v3 * b[j + 2 * ldb] + v4 * b[j + 3 * ldb];
            a[j] -= sum * tau;
            b[j] -= sum * t1;
            b[j + ldb] -= sum * t2;
            b[j + 2 * ldb] -= sum * t3;
            b[j + 3 * ldb] -= sum * t4;
        }
        return;
    }

    if (order == 6) {
        f64 v1 = v[iv];
        f64 t1 = tau * v1;
        iv += incv;
        f64 v2 = v[iv];
        f64 t2 = tau * v2;
        iv += incv;
        f64 v3 = v[iv];
        f64 t3 = tau * v3;
        iv += incv;
        f64 v4 = v[iv];
        f64 t4 = tau * v4;
        iv += incv;
        f64 v5 = v[iv];
        f64 t5 = tau * v5;
        for (i32 j = 0; j < m; j++) {
            f64 sum = a[j] + v1 * b[j] + v2 * b[j + ldb] +
                      v3 * b[j + 2 * ldb] + v4 * b[j + 3 * ldb] +
                      v5 * b[j + 4 * ldb];
            a[j] -= sum * tau;
            b[j] -= sum * t1;
            b[j + ldb] -= sum * t2;
            b[j + 2 * ldb] -= sum * t3;
            b[j + 3 * ldb] -= sum * t4;
            b[j + 4 * ldb] -= sum * t5;
        }
        return;
    }

    if (order == 7) {
        f64 v1 = v[iv];
        f64 t1 = tau * v1;
        iv += incv;
        f64 v2 = v[iv];
        f64 t2 = tau * v2;
        iv += incv;
        f64 v3 = v[iv];
        f64 t3 = tau * v3;
        iv += incv;
        f64 v4 = v[iv];
        f64 t4 = tau * v4;
        iv += incv;
        f64 v5 = v[iv];
        f64 t5 = tau * v5;
        iv += incv;
        f64 v6 = v[iv];
        f64 t6 = tau * v6;
        for (i32 j = 0; j < m; j++) {
            f64 sum = a[j] + v1 * b[j] + v2 * b[j + ldb] +
                      v3 * b[j + 2 * ldb] + v4 * b[j + 3 * ldb] +
                      v5 * b[j + 4 * ldb] + v6 * b[j + 5 * ldb];
            a[j] -= sum * tau;
            b[j] -= sum * t1;
            b[j + ldb] -= sum * t2;
            b[j + 2 * ldb] -= sum * t3;
            b[j + 3 * ldb] -= sum * t4;
            b[j + 4 * ldb] -= sum * t5;
            b[j + 5 * ldb] -= sum * t6;
        }
        return;
    }

    if (order == 8) {
        f64 v1 = v[iv];
        f64 t1 = tau * v1;
        iv += incv;
        f64 v2 = v[iv];
        f64 t2 = tau * v2;
        iv += incv;
        f64 v3 = v[iv];
        f64 t3 = tau * v3;
        iv += incv;
        f64 v4 = v[iv];
        f64 t4 = tau * v4;
        iv += incv;
        f64 v5 = v[iv];
        f64 t5 = tau * v5;
        iv += incv;
        f64 v6 = v[iv];
        f64 t6 = tau * v6;
        iv += incv;
        f64 v7 = v[iv];
        f64 t7 = tau * v7;
        for (i32 j = 0; j < m; j++) {
            f64 sum = a[j] + v1 * b[j] + v2 * b[j + ldb] +
                      v3 * b[j + 2 * ldb] + v4 * b[j + 3 * ldb] +
                      v5 * b[j + 4 * ldb] + v6 * b[j + 5 * ldb] +
                      v7 * b[j + 6 * ldb];
            a[j] -= sum * tau;
            b[j] -= sum * t1;
            b[j + ldb] -= sum * t2;
            b[j + 2 * ldb] -= sum * t3;
            b[j + 3 * ldb] -= sum * t4;
            b[j + 4 * ldb] -= sum * t5;
            b[j + 5 * ldb] -= sum * t6;
            b[j + 6 * ldb] -= sum * t7;
        }
        return;
    }

    if (order == 9) {
        f64 v1 = v[iv];
        f64 t1 = tau * v1;
        iv += incv;
        f64 v2 = v[iv];
        f64 t2 = tau * v2;
        iv += incv;
        f64 v3 = v[iv];
        f64 t3 = tau * v3;
        iv += incv;
        f64 v4 = v[iv];
        f64 t4 = tau * v4;
        iv += incv;
        f64 v5 = v[iv];
        f64 t5 = tau * v5;
        iv += incv;
        f64 v6 = v[iv];
        f64 t6 = tau * v6;
        iv += incv;
        f64 v7 = v[iv];
        f64 t7 = tau * v7;
        iv += incv;
        f64 v8 = v[iv];
        f64 t8 = tau * v8;
        for (i32 j = 0; j < m; j++) {
            f64 sum = a[j] + v1 * b[j] + v2 * b[j + ldb] +
                      v3 * b[j + 2 * ldb] + v4 * b[j + 3 * ldb] +
                      v5 * b[j + 4 * ldb] + v6 * b[j + 5 * ldb] +
                      v7 * b[j + 6 * ldb] + v8 * b[j + 7 * ldb];
            a[j] -= sum * tau;
            b[j] -= sum * t1;
            b[j + ldb] -= sum * t2;
            b[j + 2 * ldb] -= sum * t3;
            b[j + 3 * ldb] -= sum * t4;
            b[j + 4 * ldb] -= sum * t5;
            b[j + 5 * ldb] -= sum * t6;
            b[j + 6 * ldb] -= sum * t7;
            b[j + 7 * ldb] -= sum * t8;
        }
        return;
    }

    if (order == 10) {
        f64 v1 = v[iv];
        f64 t1 = tau * v1;
        iv += incv;
        f64 v2 = v[iv];
        f64 t2 = tau * v2;
        iv += incv;
        f64 v3 = v[iv];
        f64 t3 = tau * v3;
        iv += incv;
        f64 v4 = v[iv];
        f64 t4 = tau * v4;
        iv += incv;
        f64 v5 = v[iv];
        f64 t5 = tau * v5;
        iv += incv;
        f64 v6 = v[iv];
        f64 t6 = tau * v6;
        iv += incv;
        f64 v7 = v[iv];
        f64 t7 = tau * v7;
        iv += incv;
        f64 v8 = v[iv];
        f64 t8 = tau * v8;
        iv += incv;
        f64 v9 = v[iv];
        f64 t9 = tau * v9;
        for (i32 j = 0; j < m; j++) {
            f64 sum = a[j] + v1 * b[j] + v2 * b[j + ldb] +
                      v3 * b[j + 2 * ldb] + v4 * b[j + 3 * ldb] +
                      v5 * b[j + 4 * ldb] + v6 * b[j + 5 * ldb] +
                      v7 * b[j + 6 * ldb] + v8 * b[j + 7 * ldb] +
                      v9 * b[j + 8 * ldb];
            a[j] -= sum * tau;
            b[j] -= sum * t1;
            b[j + ldb] -= sum * t2;
            b[j + 2 * ldb] -= sum * t3;
            b[j + 3 * ldb] -= sum * t4;
            b[j + 4 * ldb] -= sum * t5;
            b[j + 5 * ldb] -= sum * t6;
            b[j + 6 * ldb] -= sum * t7;
            b[j + 7 * ldb] -= sum * t8;
            b[j + 8 * ldb] -= sum * t9;
        }
        return;
    }

    // General case: order >= 11, use BLAS
    // w := A (copy first column to workspace)
    // w := w + B*v = A + B*v  (this is C*u)
    // A := A - tau*w
    // B := B - tau*w*v' (rank-1 update)

    i32 inc1 = 1;
    f64 mtau = -tau;

    SLC_DCOPY(&m, a, &inc1, dwork, &inc1);
    SLC_DGEMV("N", &m, &n, &one, b, &ldb, v, &incv, &one, dwork, &inc1);
    SLC_DAXPY(&m, &mtau, dwork, &inc1, a, &inc1);
    SLC_DGER(&m, &n, &mtau, dwork, &inc1, v, &incv, b, &ldb);
}
