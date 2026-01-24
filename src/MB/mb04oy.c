// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"

void SLC_MB04OY(i32 m, i32 n, const f64* v, f64 tau,
                f64* a, i32 lda, f64* b, i32 ldb, f64* dwork) {
    const f64 zero = 0.0;
    const f64 one = 1.0;

    if (tau == zero) {
        return;
    }

    i32 order = m + 1;

    if (order == 1) {
        f64 t1 = one - tau;
        for (i32 j = 0; j < n; j++) {
            a[j * lda] *= t1;
        }
        return;
    }

    if (order == 2) {
        f64 v1 = v[0];
        f64 t1 = tau * v1;
        for (i32 j = 0; j < n; j++) {
            f64 sum = a[j * lda] + v1 * b[j * ldb];
            a[j * lda] -= sum * tau;
            b[j * ldb] -= sum * t1;
        }
        return;
    }

    if (order == 3) {
        f64 v1 = v[0];
        f64 t1 = tau * v1;
        f64 v2 = v[1];
        f64 t2 = tau * v2;
        for (i32 j = 0; j < n; j++) {
            f64 sum = a[j * lda] + v1 * b[j * ldb] + v2 * b[1 + j * ldb];
            a[j * lda] -= sum * tau;
            b[j * ldb] -= sum * t1;
            b[1 + j * ldb] -= sum * t2;
        }
        return;
    }

    if (order == 4) {
        f64 v1 = v[0];
        f64 t1 = tau * v1;
        f64 v2 = v[1];
        f64 t2 = tau * v2;
        f64 v3 = v[2];
        f64 t3 = tau * v3;
        for (i32 j = 0; j < n; j++) {
            f64 sum = a[j * lda] + v1 * b[j * ldb] + v2 * b[1 + j * ldb] +
                      v3 * b[2 + j * ldb];
            a[j * lda] -= sum * tau;
            b[j * ldb] -= sum * t1;
            b[1 + j * ldb] -= sum * t2;
            b[2 + j * ldb] -= sum * t3;
        }
        return;
    }

    if (order == 5) {
        f64 v1 = v[0];
        f64 t1 = tau * v1;
        f64 v2 = v[1];
        f64 t2 = tau * v2;
        f64 v3 = v[2];
        f64 t3 = tau * v3;
        f64 v4 = v[3];
        f64 t4 = tau * v4;
        for (i32 j = 0; j < n; j++) {
            f64 sum = a[j * lda] + v1 * b[j * ldb] + v2 * b[1 + j * ldb] +
                      v3 * b[2 + j * ldb] + v4 * b[3 + j * ldb];
            a[j * lda] -= sum * tau;
            b[j * ldb] -= sum * t1;
            b[1 + j * ldb] -= sum * t2;
            b[2 + j * ldb] -= sum * t3;
            b[3 + j * ldb] -= sum * t4;
        }
        return;
    }

    if (order == 6) {
        f64 v1 = v[0];
        f64 t1 = tau * v1;
        f64 v2 = v[1];
        f64 t2 = tau * v2;
        f64 v3 = v[2];
        f64 t3 = tau * v3;
        f64 v4 = v[3];
        f64 t4 = tau * v4;
        f64 v5 = v[4];
        f64 t5 = tau * v5;
        for (i32 j = 0; j < n; j++) {
            f64 sum = a[j * lda] + v1 * b[j * ldb] + v2 * b[1 + j * ldb] +
                      v3 * b[2 + j * ldb] + v4 * b[3 + j * ldb] +
                      v5 * b[4 + j * ldb];
            a[j * lda] -= sum * tau;
            b[j * ldb] -= sum * t1;
            b[1 + j * ldb] -= sum * t2;
            b[2 + j * ldb] -= sum * t3;
            b[3 + j * ldb] -= sum * t4;
            b[4 + j * ldb] -= sum * t5;
        }
        return;
    }

    if (order == 7) {
        f64 v1 = v[0];
        f64 t1 = tau * v1;
        f64 v2 = v[1];
        f64 t2 = tau * v2;
        f64 v3 = v[2];
        f64 t3 = tau * v3;
        f64 v4 = v[3];
        f64 t4 = tau * v4;
        f64 v5 = v[4];
        f64 t5 = tau * v5;
        f64 v6 = v[5];
        f64 t6 = tau * v6;
        for (i32 j = 0; j < n; j++) {
            f64 sum = a[j * lda] + v1 * b[j * ldb] + v2 * b[1 + j * ldb] +
                      v3 * b[2 + j * ldb] + v4 * b[3 + j * ldb] +
                      v5 * b[4 + j * ldb] + v6 * b[5 + j * ldb];
            a[j * lda] -= sum * tau;
            b[j * ldb] -= sum * t1;
            b[1 + j * ldb] -= sum * t2;
            b[2 + j * ldb] -= sum * t3;
            b[3 + j * ldb] -= sum * t4;
            b[4 + j * ldb] -= sum * t5;
            b[5 + j * ldb] -= sum * t6;
        }
        return;
    }

    if (order == 8) {
        f64 v1 = v[0];
        f64 t1 = tau * v1;
        f64 v2 = v[1];
        f64 t2 = tau * v2;
        f64 v3 = v[2];
        f64 t3 = tau * v3;
        f64 v4 = v[3];
        f64 t4 = tau * v4;
        f64 v5 = v[4];
        f64 t5 = tau * v5;
        f64 v6 = v[5];
        f64 t6 = tau * v6;
        f64 v7 = v[6];
        f64 t7 = tau * v7;
        for (i32 j = 0; j < n; j++) {
            f64 sum = a[j * lda] + v1 * b[j * ldb] + v2 * b[1 + j * ldb] +
                      v3 * b[2 + j * ldb] + v4 * b[3 + j * ldb] +
                      v5 * b[4 + j * ldb] + v6 * b[5 + j * ldb] +
                      v7 * b[6 + j * ldb];
            a[j * lda] -= sum * tau;
            b[j * ldb] -= sum * t1;
            b[1 + j * ldb] -= sum * t2;
            b[2 + j * ldb] -= sum * t3;
            b[3 + j * ldb] -= sum * t4;
            b[4 + j * ldb] -= sum * t5;
            b[5 + j * ldb] -= sum * t6;
            b[6 + j * ldb] -= sum * t7;
        }
        return;
    }

    if (order == 9) {
        f64 v1 = v[0];
        f64 t1 = tau * v1;
        f64 v2 = v[1];
        f64 t2 = tau * v2;
        f64 v3 = v[2];
        f64 t3 = tau * v3;
        f64 v4 = v[3];
        f64 t4 = tau * v4;
        f64 v5 = v[4];
        f64 t5 = tau * v5;
        f64 v6 = v[5];
        f64 t6 = tau * v6;
        f64 v7 = v[6];
        f64 t7 = tau * v7;
        f64 v8 = v[7];
        f64 t8 = tau * v8;
        for (i32 j = 0; j < n; j++) {
            f64 sum = a[j * lda] + v1 * b[j * ldb] + v2 * b[1 + j * ldb] +
                      v3 * b[2 + j * ldb] + v4 * b[3 + j * ldb] +
                      v5 * b[4 + j * ldb] + v6 * b[5 + j * ldb] +
                      v7 * b[6 + j * ldb] + v8 * b[7 + j * ldb];
            a[j * lda] -= sum * tau;
            b[j * ldb] -= sum * t1;
            b[1 + j * ldb] -= sum * t2;
            b[2 + j * ldb] -= sum * t3;
            b[3 + j * ldb] -= sum * t4;
            b[4 + j * ldb] -= sum * t5;
            b[5 + j * ldb] -= sum * t6;
            b[6 + j * ldb] -= sum * t7;
            b[7 + j * ldb] -= sum * t8;
        }
        return;
    }

    if (order == 10) {
        f64 v1 = v[0];
        f64 t1 = tau * v1;
        f64 v2 = v[1];
        f64 t2 = tau * v2;
        f64 v3 = v[2];
        f64 t3 = tau * v3;
        f64 v4 = v[3];
        f64 t4 = tau * v4;
        f64 v5 = v[4];
        f64 t5 = tau * v5;
        f64 v6 = v[5];
        f64 t6 = tau * v6;
        f64 v7 = v[6];
        f64 t7 = tau * v7;
        f64 v8 = v[7];
        f64 t8 = tau * v8;
        f64 v9 = v[8];
        f64 t9 = tau * v9;
        for (i32 j = 0; j < n; j++) {
            f64 sum = a[j * lda] + v1 * b[j * ldb] + v2 * b[1 + j * ldb] +
                      v3 * b[2 + j * ldb] + v4 * b[3 + j * ldb] +
                      v5 * b[4 + j * ldb] + v6 * b[5 + j * ldb] +
                      v7 * b[6 + j * ldb] + v8 * b[7 + j * ldb] +
                      v9 * b[8 + j * ldb];
            a[j * lda] -= sum * tau;
            b[j * ldb] -= sum * t1;
            b[1 + j * ldb] -= sum * t2;
            b[2 + j * ldb] -= sum * t3;
            b[3 + j * ldb] -= sum * t4;
            b[4 + j * ldb] -= sum * t5;
            b[5 + j * ldb] -= sum * t6;
            b[6 + j * ldb] -= sum * t7;
            b[7 + j * ldb] -= sum * t8;
            b[8 + j * ldb] -= sum * t9;
        }
        return;
    }

    i32 inc1 = 1;
    f64 mtau = -tau;

    SLC_DCOPY(&n, a, &lda, dwork, &inc1);
    SLC_DGEMV("T", &m, &n, &one, b, &ldb, v, &inc1, &one, dwork, &inc1);
    SLC_DAXPY(&n, &mtau, dwork, &inc1, a, &lda);
    SLC_DGER(&m, &n, &mtau, v, &inc1, dwork, &inc1, b, &ldb);
}

void mb04oy(const i32* m, const i32* n, const f64* v, const f64* tau,
            f64* c1, const i32* ldc1, f64* c2, const i32* ldc2, f64* dwork) {
    SLC_MB04OY(*m, *n, v, *tau, c1, *ldc1, c2, *ldc2, dwork);
}
