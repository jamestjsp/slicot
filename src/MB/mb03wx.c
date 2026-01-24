/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB03WX - Compute eigenvalues of a product of matrices in periodic Schur form
 *
 * Computes eigenvalues of T = T_1*T_2*...*T_p where:
 * - T_1 is upper quasi-triangular (real Schur form)
 * - T_2, ..., T_p are upper triangular
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb03wx(const i32 n, const i32 p, const f64* t, const i32 ldt1, const i32 ldt2,
            f64* wr, f64* wi, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    i32 i, i1, inext, j;
    f64 a11, a12, a21, a22, cs, sn, t11, t12, t22;

    *info = 0;

    // Parameter validation
    if (n < 0) {
        *info = -1;
        return;
    }
    if (p < 1) {
        *info = -2;
        return;
    }
    if (ldt1 < (n > 1 ? n : 1)) {
        *info = -4;
        return;
    }
    if (ldt2 < (n > 1 ? n : 1)) {
        *info = -5;
        return;
    }

    // Quick return for n = 0
    if (n == 0) {
        return;
    }

    // 3D array indexing: T(i,j,k) in Fortran = t[i + j*ldt1 + k*ldt1*ldt2] in C (0-based)
    // Fortran 1-based: T(I,J,K) -> t[(i-1) + (j-1)*ldt1 + (k-1)*ldt1*ldt2]
    #define T(i, j, k) t[(i) + (j)*ldt1 + (k)*ldt1*ldt2]

    inext = 0;
    for (i = 0; i < n; i++) {
        if (i < inext) {
            continue;
        }

        if (i != n - 1) {
            // Check for 2x2 block (subdiagonal element non-zero in T_1)
            if (T(i + 1, i, 0) != zero) {
                // A pair of eigenvalues. First compute the corresponding
                // elements of T(i:i+1,i:i+1).
                inext = i + 2;
                i1 = i + 1;
                t11 = one;
                t12 = zero;
                t22 = one;

                // Accumulate product for j = 2 to p (indices 1 to p-1 in C)
                for (j = 1; j < p; j++) {
                    t22 = t22 * T(i1, i1, j);
                    t12 = t11 * T(i, i1, j) + t12 * T(i1, i1, j);
                    t11 = t11 * T(i, i, j);
                }

                // Form 2x2 product matrix
                a11 = T(i, i, 0) * t11;
                a12 = T(i, i, 0) * t12 + T(i, i1, 0) * t22;
                a21 = T(i1, i, 0) * t11;
                a22 = T(i1, i, 0) * t12 + T(i1, i1, 0) * t22;

                // Compute eigenvalues of 2x2 matrix using DLANV2
                SLC_DLANV2(&a11, &a12, &a21, &a22, &wr[i], &wi[i],
                           &wr[i1], &wi[i1], &cs, &sn);
                continue;
            }
        }

        // Simple eigenvalue. Compute the corresponding element of T(I,I).
        inext = i + 1;
        t11 = one;

        for (j = 0; j < p; j++) {
            t11 = t11 * T(i, i, j);
        }

        wr[i] = t11;
        wi[i] = zero;
    }

    #undef T
}
