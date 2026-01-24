/**
 * @file mc03md.c
 * @brief Compute polynomial matrix operation P(x) = P1(x) * P2(x) + alpha * P3(x)
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"

void SLC_MC03MD(
    i32 rp1, i32 cp1, i32 cp2,
    i32 dp1, i32 dp2, i32 *dp3,
    f64 alpha,
    const f64 *p1, i32 ldp11, i32 ldp12,
    const f64 *p2, i32 ldp21, i32 ldp22,
    f64 *p3, i32 ldp31, i32 ldp32,
    f64 *dwork,
    i32 *info)
{
    const f64 zero = 0.0;
    i32 dpol3, e, h, i, j, k;
    bool cfzero;
    i32 int1 = 1;

    *info = 0;

    // Validate input parameters
    if (rp1 < 0) {
        *info = -1;
    } else if (cp1 < 0) {
        *info = -2;
    } else if (cp2 < 0) {
        *info = -3;
    } else if (dp1 < -1) {
        *info = -4;
    } else if (dp2 < -1) {
        *info = -5;
    } else if (*dp3 < -1) {
        *info = -6;
    } else if ((dp1 == -1 && ldp11 < 1) ||
               (dp1 >= 0 && ldp11 < (rp1 > 1 ? rp1 : 1))) {
        *info = -9;
    } else if ((dp1 == -1 && ldp12 < 1) ||
               (dp1 >= 0 && ldp12 < (cp1 > 1 ? cp1 : 1))) {
        *info = -10;
    } else if ((dp2 == -1 && ldp21 < 1) ||
               (dp2 >= 0 && ldp21 < (cp1 > 1 ? cp1 : 1))) {
        *info = -12;
    } else if ((dp2 == -1 && ldp22 < 1) ||
               (dp2 >= 0 && ldp22 < (cp2 > 1 ? cp2 : 1))) {
        *info = -13;
    } else if (ldp31 < (rp1 > 1 ? rp1 : 1)) {
        *info = -15;
    } else if (ldp32 < (cp2 > 1 ? cp2 : 1)) {
        *info = -16;
    }

    if (*info != 0) {
        return;
    }

    // Quick return if possible
    if (rp1 == 0 || cp2 == 0) {
        return;
    }

    if (alpha == zero) {
        *dp3 = -1;
    }

    // Scale P3 by alpha if P3 is non-zero
    if (*dp3 >= 0) {
        for (k = 0; k <= *dp3; k++) {
            for (j = 0; j < cp2; j++) {
                // P3(1:rp1, j, k) *= alpha
                // 3D array: P3(i,j,k) = p3[i + j*ldp31 + k*ldp31*ldp32]
                f64 *col = &p3[j * ldp31 + k * ldp31 * ldp32];
                SLC_DSCAL(&rp1, &alpha, col, &int1);
            }
        }
    }

    // If P1 or P2 is zero, return
    if (dp1 == -1 || dp2 == -1 || cp1 == 0) {
        return;
    }

    // Neither P1 nor P2 is zero polynomial
    dpol3 = dp1 + dp2;

    // Initialize additional part of P3 to zero if needed
    if (dpol3 > *dp3) {
        for (k = *dp3 + 1; k <= dpol3; k++) {
            // Set P3(:,:,k) to zero
            for (j = 0; j < cp2; j++) {
                for (i = 0; i < rp1; i++) {
                    p3[i + j * ldp31 + k * ldp31 * ldp32] = zero;
                }
            }
        }
        *dp3 = dpol3;
    }

    // Compute polynomial matrix product
    // The inner product of the j-th row of coefficient k-1 of P1
    // and the h-th column of coefficient i-1 of P2 contributes
    // to the (j,h)-th element of coefficient k+i-2 of P3
    for (k = 0; k <= dp1; k++) {
        for (j = 0; j < rp1; j++) {
            // Copy j-th row of P1(:,:,k) to dwork
            // P1(j, 0:cp1-1, k) - row j of coefficient matrix k
            for (i32 col = 0; col < cp1; col++) {
                dwork[col] = p1[j + col * ldp11 + k * ldp11 * ldp12];
            }

            for (i = 0; i <= dp2; i++) {
                e = k + i;  // Combined degree index (0-based)

                for (h = 0; h < cp2; h++) {
                    // P3(j,h,e) += dot(dwork, P2(:,h,i))
                    // P2(:,h,i) is column h of coefficient matrix i
                    const f64 *p2_col = &p2[h * ldp21 + i * ldp21 * ldp22];
                    f64 dot_result = SLC_DDOT(&cp1, dwork, &int1, p2_col, &int1);
                    p3[j + h * ldp31 + e * ldp31 * ldp32] += dot_result;
                }
            }
        }
    }

    // Compute exact degree of result P3
    cfzero = true;
    while (*dp3 >= 0 && cfzero) {
        dpol3 = *dp3;
        for (j = 0; j < cp2 && cfzero; j++) {
            for (i = 0; i < rp1 && cfzero; i++) {
                if (p3[i + j * ldp31 + dpol3 * ldp31 * ldp32] != zero) {
                    cfzero = false;
                }
            }
        }
        if (cfzero) {
            (*dp3)--;
        }
    }
}
