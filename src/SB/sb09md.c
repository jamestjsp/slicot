/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB09MD - Evaluation of closeness of two multivariable sequences
 *
 * Compares two multivariable sequences M1(k) and M2(k) for k = 1,2,...,N
 * and evaluates their closeness. Each M1(k) and M2(k) is an NC by NB matrix.
 *
 * Computes:
 * - SS(i,j) = sum_{k=1}^{N} M1(i,j,k)^2
 * - SE(i,j) = sum_{k=1}^{N} (M1(i,j,k) - M2(i,j,k))^2
 * - PRE(i,j) = 100 * sqrt(SE(i,j) / SS(i,j))
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb09md(
    const i32 n,
    const i32 nc,
    const i32 nb,
    const f64* h1,
    const i32 ldh1,
    const f64* h2,
    const i32 ldh2,
    f64* ss,
    const i32 ldss,
    f64* se,
    const i32 ldse,
    f64* pre,
    const i32 ldpre,
    const f64 tol,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 hundrd = 100.0;

    *info = 0;

    // Parameter validation
    i32 minld = (nc > 1) ? nc : 1;
    if (n < 0) {
        *info = -1;
    } else if (nc < 0) {
        *info = -2;
    } else if (nb < 0) {
        *info = -3;
    } else if (ldh1 < minld) {
        *info = -5;
    } else if (ldh2 < minld) {
        *info = -7;
    } else if (ldss < minld) {
        *info = -9;
    } else if (ldse < minld) {
        *info = -11;
    } else if (ldpre < minld) {
        *info = -13;
    }

    if (*info != 0) {
        return;
    }

    // Quick return if possible
    if (n == 0 || nc == 0 || nb == 0) {
        return;
    }

    f64 eps = SLC_DLAMCH("Epsilon");
    f64 toler = (tol > eps) ? tol : eps;
    f64 epso = one / toler;

    for (i32 j = 0; j < nb; j++) {
        for (i32 i = 0; i < nc; i++) {
            f64 sse = zero;
            f64 sss = zero;
            bool noflow = true;
            i32 k = 0;

            // WHILE ( ( NOFLOW .AND. ( K .LT. N*NB ) ) DO
            while (noflow && k < n * nb) {
                // H1(I,K+J) in Fortran (1-based) -> h1[i + (k+j)*ldh1] in C
                // But k increments by nb, so column index is (k + j)
                // k=0: col j; k=nb: col nb+j; k=2*nb: col 2*nb+j
                f64 var = h1[i + (k + j) * ldh1];
                f64 vare = h2[i + (k + j) * ldh2] - var;

                if (fabs(var) > epso || fabs(vare) > epso) {
                    se[i + j * ldse] = epso;
                    ss[i + j * ldss] = epso;
                    pre[i + j * ldpre] = one;
                    noflow = false;
                } else {
                    if (fabs(vare) > toler) {
                        sse += vare * vare;
                    }
                    if (fabs(var) > toler) {
                        sss += var * var;
                    }
                    k += nb;
                }
            }

            if (noflow) {
                se[i + j * ldse] = sse;
                ss[i + j * ldss] = sss;
                pre[i + j * ldpre] = hundrd;
                if (sss > toler) {
                    pre[i + j * ldpre] = sqrt(sse / sss) * hundrd;
                }
            }
        }
    }
}
