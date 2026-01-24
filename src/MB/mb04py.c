// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"

#include <stdbool.h>

void SLC_MB04PY(char side, i32 m, i32 n, const f64* v, f64 tau,
                f64* c, i32 ldc, f64* dwork) {
    const f64 zero = 0.0;
    const f64 one = 1.0;

    if (tau == zero) {
        return;
    }

    bool left = (side == 'L' || side == 'l');

    if (left) {
        switch (m) {
        case 1: {
            f64 t1 = one - tau;
            for (i32 j = 0; j < n; j++) {
                c[j * ldc] *= t1;
            }
            return;
        }
        case 2: {
            f64 v1 = v[0];
            f64 t1 = tau * v1;
            for (i32 j = 0; j < n; j++) {
                f64 sum = c[j * ldc] + v1 * c[1 + j * ldc];
                c[j * ldc] -= sum * tau;
                c[1 + j * ldc] -= sum * t1;
            }
            return;
        }
        case 3: {
            f64 v1 = v[0];
            f64 t1 = tau * v1;
            f64 v2 = v[1];
            f64 t2 = tau * v2;
            for (i32 j = 0; j < n; j++) {
                f64 sum = c[j * ldc] + v1 * c[1 + j * ldc] + v2 * c[2 + j * ldc];
                c[j * ldc] -= sum * tau;
                c[1 + j * ldc] -= sum * t1;
                c[2 + j * ldc] -= sum * t2;
            }
            return;
        }
        case 4: {
            f64 v1 = v[0];
            f64 t1 = tau * v1;
            f64 v2 = v[1];
            f64 t2 = tau * v2;
            f64 v3 = v[2];
            f64 t3 = tau * v3;
            for (i32 j = 0; j < n; j++) {
                f64 sum = c[j * ldc] + v1 * c[1 + j * ldc] + v2 * c[2 + j * ldc] +
                          v3 * c[3 + j * ldc];
                c[j * ldc] -= sum * tau;
                c[1 + j * ldc] -= sum * t1;
                c[2 + j * ldc] -= sum * t2;
                c[3 + j * ldc] -= sum * t3;
            }
            return;
        }
        case 5: {
            f64 v1 = v[0];
            f64 t1 = tau * v1;
            f64 v2 = v[1];
            f64 t2 = tau * v2;
            f64 v3 = v[2];
            f64 t3 = tau * v3;
            f64 v4 = v[3];
            f64 t4 = tau * v4;
            for (i32 j = 0; j < n; j++) {
                f64 sum = c[j * ldc] + v1 * c[1 + j * ldc] + v2 * c[2 + j * ldc] +
                          v3 * c[3 + j * ldc] + v4 * c[4 + j * ldc];
                c[j * ldc] -= sum * tau;
                c[1 + j * ldc] -= sum * t1;
                c[2 + j * ldc] -= sum * t2;
                c[3 + j * ldc] -= sum * t3;
                c[4 + j * ldc] -= sum * t4;
            }
            return;
        }
        case 6: {
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
                f64 sum = c[j * ldc] + v1 * c[1 + j * ldc] + v2 * c[2 + j * ldc] +
                          v3 * c[3 + j * ldc] + v4 * c[4 + j * ldc] +
                          v5 * c[5 + j * ldc];
                c[j * ldc] -= sum * tau;
                c[1 + j * ldc] -= sum * t1;
                c[2 + j * ldc] -= sum * t2;
                c[3 + j * ldc] -= sum * t3;
                c[4 + j * ldc] -= sum * t4;
                c[5 + j * ldc] -= sum * t5;
            }
            return;
        }
        case 7: {
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
                f64 sum = c[j * ldc] + v1 * c[1 + j * ldc] + v2 * c[2 + j * ldc] +
                          v3 * c[3 + j * ldc] + v4 * c[4 + j * ldc] +
                          v5 * c[5 + j * ldc] + v6 * c[6 + j * ldc];
                c[j * ldc] -= sum * tau;
                c[1 + j * ldc] -= sum * t1;
                c[2 + j * ldc] -= sum * t2;
                c[3 + j * ldc] -= sum * t3;
                c[4 + j * ldc] -= sum * t4;
                c[5 + j * ldc] -= sum * t5;
                c[6 + j * ldc] -= sum * t6;
            }
            return;
        }
        case 8: {
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
                f64 sum = c[j * ldc] + v1 * c[1 + j * ldc] + v2 * c[2 + j * ldc] +
                          v3 * c[3 + j * ldc] + v4 * c[4 + j * ldc] +
                          v5 * c[5 + j * ldc] + v6 * c[6 + j * ldc] +
                          v7 * c[7 + j * ldc];
                c[j * ldc] -= sum * tau;
                c[1 + j * ldc] -= sum * t1;
                c[2 + j * ldc] -= sum * t2;
                c[3 + j * ldc] -= sum * t3;
                c[4 + j * ldc] -= sum * t4;
                c[5 + j * ldc] -= sum * t5;
                c[6 + j * ldc] -= sum * t6;
                c[7 + j * ldc] -= sum * t7;
            }
            return;
        }
        case 9: {
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
                f64 sum = c[j * ldc] + v1 * c[1 + j * ldc] + v2 * c[2 + j * ldc] +
                          v3 * c[3 + j * ldc] + v4 * c[4 + j * ldc] +
                          v5 * c[5 + j * ldc] + v6 * c[6 + j * ldc] +
                          v7 * c[7 + j * ldc] + v8 * c[8 + j * ldc];
                c[j * ldc] -= sum * tau;
                c[1 + j * ldc] -= sum * t1;
                c[2 + j * ldc] -= sum * t2;
                c[3 + j * ldc] -= sum * t3;
                c[4 + j * ldc] -= sum * t4;
                c[5 + j * ldc] -= sum * t5;
                c[6 + j * ldc] -= sum * t6;
                c[7 + j * ldc] -= sum * t7;
                c[8 + j * ldc] -= sum * t8;
            }
            return;
        }
        case 10: {
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
                f64 sum = c[j * ldc] + v1 * c[1 + j * ldc] + v2 * c[2 + j * ldc] +
                          v3 * c[3 + j * ldc] + v4 * c[4 + j * ldc] +
                          v5 * c[5 + j * ldc] + v6 * c[6 + j * ldc] +
                          v7 * c[7 + j * ldc] + v8 * c[8 + j * ldc] +
                          v9 * c[9 + j * ldc];
                c[j * ldc] -= sum * tau;
                c[1 + j * ldc] -= sum * t1;
                c[2 + j * ldc] -= sum * t2;
                c[3 + j * ldc] -= sum * t3;
                c[4 + j * ldc] -= sum * t4;
                c[5 + j * ldc] -= sum * t5;
                c[6 + j * ldc] -= sum * t6;
                c[7 + j * ldc] -= sum * t7;
                c[8 + j * ldc] -= sum * t8;
                c[9 + j * ldc] -= sum * t9;
            }
            return;
        }
        default:
            break;
        }

        i32 inc1 = 1;
        i32 m1 = m - 1;
        f64 mtau = -tau;

        SLC_DCOPY(&n, c, &ldc, dwork, &inc1);
        SLC_DGEMV("T", &m1, &n, &one, &c[1], &ldc, v, &inc1, &one, dwork, &inc1);
        SLC_DAXPY(&n, &mtau, dwork, &inc1, c, &ldc);
        SLC_DGER(&m1, &n, &mtau, v, &inc1, dwork, &inc1, &c[1], &ldc);
    } else {
        switch (n) {
        case 1: {
            f64 t1 = one - tau;
            for (i32 j = 0; j < m; j++) {
                c[j] *= t1;
            }
            return;
        }
        case 2: {
            f64 v1 = v[0];
            f64 t1 = tau * v1;
            for (i32 j = 0; j < m; j++) {
                f64 sum = c[j] + v1 * c[j + ldc];
                c[j] -= sum * tau;
                c[j + ldc] -= sum * t1;
            }
            return;
        }
        case 3: {
            f64 v1 = v[0];
            f64 t1 = tau * v1;
            f64 v2 = v[1];
            f64 t2 = tau * v2;
            for (i32 j = 0; j < m; j++) {
                f64 sum = c[j] + v1 * c[j + ldc] + v2 * c[j + 2 * ldc];
                c[j] -= sum * tau;
                c[j + ldc] -= sum * t1;
                c[j + 2 * ldc] -= sum * t2;
            }
            return;
        }
        case 4: {
            f64 v1 = v[0];
            f64 t1 = tau * v1;
            f64 v2 = v[1];
            f64 t2 = tau * v2;
            f64 v3 = v[2];
            f64 t3 = tau * v3;
            for (i32 j = 0; j < m; j++) {
                f64 sum = c[j] + v1 * c[j + ldc] + v2 * c[j + 2 * ldc] +
                          v3 * c[j + 3 * ldc];
                c[j] -= sum * tau;
                c[j + ldc] -= sum * t1;
                c[j + 2 * ldc] -= sum * t2;
                c[j + 3 * ldc] -= sum * t3;
            }
            return;
        }
        case 5: {
            f64 v1 = v[0];
            f64 t1 = tau * v1;
            f64 v2 = v[1];
            f64 t2 = tau * v2;
            f64 v3 = v[2];
            f64 t3 = tau * v3;
            f64 v4 = v[3];
            f64 t4 = tau * v4;
            for (i32 j = 0; j < m; j++) {
                f64 sum = c[j] + v1 * c[j + ldc] + v2 * c[j + 2 * ldc] +
                          v3 * c[j + 3 * ldc] + v4 * c[j + 4 * ldc];
                c[j] -= sum * tau;
                c[j + ldc] -= sum * t1;
                c[j + 2 * ldc] -= sum * t2;
                c[j + 3 * ldc] -= sum * t3;
                c[j + 4 * ldc] -= sum * t4;
            }
            return;
        }
        case 6: {
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
            for (i32 j = 0; j < m; j++) {
                f64 sum = c[j] + v1 * c[j + ldc] + v2 * c[j + 2 * ldc] +
                          v3 * c[j + 3 * ldc] + v4 * c[j + 4 * ldc] +
                          v5 * c[j + 5 * ldc];
                c[j] -= sum * tau;
                c[j + ldc] -= sum * t1;
                c[j + 2 * ldc] -= sum * t2;
                c[j + 3 * ldc] -= sum * t3;
                c[j + 4 * ldc] -= sum * t4;
                c[j + 5 * ldc] -= sum * t5;
            }
            return;
        }
        case 7: {
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
            for (i32 j = 0; j < m; j++) {
                f64 sum = c[j] + v1 * c[j + ldc] + v2 * c[j + 2 * ldc] +
                          v3 * c[j + 3 * ldc] + v4 * c[j + 4 * ldc] +
                          v5 * c[j + 5 * ldc] + v6 * c[j + 6 * ldc];
                c[j] -= sum * tau;
                c[j + ldc] -= sum * t1;
                c[j + 2 * ldc] -= sum * t2;
                c[j + 3 * ldc] -= sum * t3;
                c[j + 4 * ldc] -= sum * t4;
                c[j + 5 * ldc] -= sum * t5;
                c[j + 6 * ldc] -= sum * t6;
            }
            return;
        }
        case 8: {
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
            for (i32 j = 0; j < m; j++) {
                f64 sum = c[j] + v1 * c[j + ldc] + v2 * c[j + 2 * ldc] +
                          v3 * c[j + 3 * ldc] + v4 * c[j + 4 * ldc] +
                          v5 * c[j + 5 * ldc] + v6 * c[j + 6 * ldc] +
                          v7 * c[j + 7 * ldc];
                c[j] -= sum * tau;
                c[j + ldc] -= sum * t1;
                c[j + 2 * ldc] -= sum * t2;
                c[j + 3 * ldc] -= sum * t3;
                c[j + 4 * ldc] -= sum * t4;
                c[j + 5 * ldc] -= sum * t5;
                c[j + 6 * ldc] -= sum * t6;
                c[j + 7 * ldc] -= sum * t7;
            }
            return;
        }
        case 9: {
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
            for (i32 j = 0; j < m; j++) {
                f64 sum = c[j] + v1 * c[j + ldc] + v2 * c[j + 2 * ldc] +
                          v3 * c[j + 3 * ldc] + v4 * c[j + 4 * ldc] +
                          v5 * c[j + 5 * ldc] + v6 * c[j + 6 * ldc] +
                          v7 * c[j + 7 * ldc] + v8 * c[j + 8 * ldc];
                c[j] -= sum * tau;
                c[j + ldc] -= sum * t1;
                c[j + 2 * ldc] -= sum * t2;
                c[j + 3 * ldc] -= sum * t3;
                c[j + 4 * ldc] -= sum * t4;
                c[j + 5 * ldc] -= sum * t5;
                c[j + 6 * ldc] -= sum * t6;
                c[j + 7 * ldc] -= sum * t7;
                c[j + 8 * ldc] -= sum * t8;
            }
            return;
        }
        case 10: {
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
            for (i32 j = 0; j < m; j++) {
                f64 sum = c[j] + v1 * c[j + ldc] + v2 * c[j + 2 * ldc] +
                          v3 * c[j + 3 * ldc] + v4 * c[j + 4 * ldc] +
                          v5 * c[j + 5 * ldc] + v6 * c[j + 6 * ldc] +
                          v7 * c[j + 7 * ldc] + v8 * c[j + 8 * ldc] +
                          v9 * c[j + 9 * ldc];
                c[j] -= sum * tau;
                c[j + ldc] -= sum * t1;
                c[j + 2 * ldc] -= sum * t2;
                c[j + 3 * ldc] -= sum * t3;
                c[j + 4 * ldc] -= sum * t4;
                c[j + 5 * ldc] -= sum * t5;
                c[j + 6 * ldc] -= sum * t6;
                c[j + 7 * ldc] -= sum * t7;
                c[j + 8 * ldc] -= sum * t8;
                c[j + 9 * ldc] -= sum * t9;
            }
            return;
        }
        default:
            break;
        }

        i32 inc1 = 1;
        i32 n1 = n - 1;
        f64 mtau = -tau;

        SLC_DCOPY(&m, c, &inc1, dwork, &inc1);
        SLC_DGEMV("N", &m, &n1, &one, &c[ldc], &ldc, v, &inc1, &one, dwork, &inc1);
        SLC_DAXPY(&m, &mtau, dwork, &inc1, c, &inc1);
        SLC_DGER(&m, &n1, &mtau, dwork, &inc1, v, &inc1, &c[ldc], &ldc);
    }
}

void mb04py(const char* side, const i32* m, const i32* n, const f64* v,
            const f64* tau, f64* c, const i32* ldc, f64* dwork) {
    SLC_MB04PY(*side, *m, *n, v, *tau, c, *ldc, dwork);
}
