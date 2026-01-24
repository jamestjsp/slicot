/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10JD - Convert descriptor state-space to regular state-space
 *
 * Converts the descriptor state-space system:
 *   E*dx/dt = A*x + B*u
 *        y = C*x + D*u
 *
 * into regular state-space form:
 *   dx/dt = Ad*x + Bd*u
 *       y = Cd*x + Dd*u
 *
 * Uses SVD decomposition of E for descriptor elimination.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

void sb10jd(
    const i32 n,
    const i32 m,
    const i32 np,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    f64* e,
    const i32 lde,
    i32* nsys,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 mone = -1.0;
    const i32 int1 = 1;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (np < 0) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldc < (np > 1 ? np : 1)) {
        *info = -9;
    } else if (ldd < (np > 1 ? np : 1)) {
        *info = -11;
    } else if (lde < (n > 1 ? n : 1)) {
        *info = -13;
    }

    i32 tmp1 = n + m + np;
    i32 tmp2 = tmp1 > 5 ? tmp1 : 5;
    i32 minwrk = 2 * n * (n + 1) + n * tmp2;
    if (minwrk < 1) minwrk = 1;

    if (ldwork < minwrk) {
        *info = -16;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        *nsys = 0;
        dwork[0] = one;
        return;
    }

    f64 eps = SLC_DLAMCH("Epsilon");
    f64 tol = sqrt(eps);

    i32 is = 0;
    i32 iu = is + n;
    i32 iv = iu + n * n;
    i32 iwrk = iv + n * n;

    i32 info2 = 0;
    i32 lwork = ldwork - iwrk;

    SLC_DGESVD("S", "S", &n, &n, e, &lde, &dwork[is], &dwork[iu], &n,
               &dwork[iv], &n, &dwork[iwrk], &lwork, &info2);

    if (info2 != 0) {
        *info = 1;
        return;
    }

    i32 lwamax = minwrk > (i32)(dwork[iwrk] + iwrk) ? minwrk : (i32)(dwork[iwrk] + iwrk);

    i32 ns1 = 0;
    for (i32 i = 0; i < n; i++) {
        if (dwork[is + i] > tol) {
            ns1++;
        }
    }

    if (ns1 > 0) {
        i32 ldnp = (np > 0) ? np : 1;

        SLC_DGEMM("T", "N", &n, &n, &n, &one, &dwork[iu], &n, a, &lda,
                  &zero, &dwork[iwrk], &n);
        SLC_DGEMM("N", "T", &n, &n, &n, &one, &dwork[iwrk], &n,
                  &dwork[iv], &n, &zero, a, &lda);

        SLC_DLACPY("Full", &n, &m, b, &ldb, &dwork[iwrk], &n);
        SLC_DGEMM("T", "N", &n, &m, &n, &one, &dwork[iu], &n,
                  &dwork[iwrk], &n, &zero, b, &ldb);

        SLC_DLACPY("Full", &np, &n, c, &ldc, &dwork[iwrk], &ldnp);
        SLC_DGEMM("N", "T", &np, &n, &n, &one, &dwork[iwrk], &ldnp,
                  &dwork[iv], &n, &zero, c, &ldc);

        i32 k = n - ns1;
        if (k > 0) {
            i32 isa = iu + k * k;
            iv = isa + k;
            i32 ivk = iv + k * ((k > ns1) ? k : ns1);
            iwrk = ivk;

            SLC_DGESVD("S", "S", &k, &k, &a[ns1 + ns1 * lda], &lda,
                       &dwork[isa], &dwork[iu], &k, &dwork[iv], &k,
                       &dwork[iwrk], &lwork, &info2);

            if (info2 != 0) {
                *info = 1;
                return;
            }

            i32 ia12 = iwrk;
            i32 ib2 = ia12 + ns1 * k;
            i32 ic2 = ib2 + k * m;

            i32 lwa = (i32)dwork[iwrk] + iwrk;
            i32 ic2_end = ic2 + k * np;
            lwamax = (lwa > lwamax) ? lwa : lwamax;
            lwamax = (ic2_end > lwamax) ? ic2_end : lwamax;

            SLC_DGEMM("N", "T", &ns1, &k, &k, &one, &a[ns1 * lda], &lda,
                      &dwork[iv], &k, &zero, &dwork[ia12], &ns1);

            SLC_DGEMM("N", "T", &np, &k, &k, &one, &c[ns1 * ldc], &ldc,
                      &dwork[iv], &k, &zero, &dwork[ic2], &ldnp);

            i32 ia21 = iv;
            SLC_DGEMM("T", "N", &k, &ns1, &k, &one, &dwork[iu], &k,
                      &a[ns1], &lda, &zero, &dwork[ia21], &k);

            SLC_DGEMM("T", "N", &k, &m, &k, &one, &dwork[iu], &k,
                      &b[ns1], &ldb, &zero, &dwork[ib2], &k);

            for (i32 j = 0; j < k; j++) {
                f64 scale = zero;
                if (dwork[isa + j] > tol) {
                    scale = one / dwork[isa + j];
                }
                SLC_DSCAL(&ns1, &scale, &dwork[ia12 + j * ns1], &int1);
                SLC_DSCAL(&np, &scale, &dwork[ic2 + j * np], &int1);
            }

            SLC_DGEMM("N", "N", &ns1, &ns1, &k, &mone, &dwork[ia12], &ns1,
                      &dwork[ia21], &k, &one, a, &lda);

            SLC_DGEMM("N", "N", &ns1, &m, &k, &mone, &dwork[ia12], &ns1,
                      &dwork[ib2], &k, &one, b, &ldb);

            SLC_DGEMM("N", "N", &np, &ns1, &k, &mone, &dwork[ic2], &ldnp,
                      &dwork[ia21], &k, &one, c, &ldc);

            SLC_DGEMM("N", "N", &np, &m, &k, &mone, &dwork[ic2], &ldnp,
                      &dwork[ib2], &k, &one, d, &ldd);
        }

        for (i32 i = 0; i < ns1; i++) {
            f64 scale = one / sqrt(dwork[is + i]);
            SLC_DSCAL(&ns1, &scale, &a[i], &lda);
            SLC_DSCAL(&m, &scale, &b[i], &ldb);
        }

        for (i32 j = 0; j < ns1; j++) {
            f64 scale = one / sqrt(dwork[is + j]);
            SLC_DSCAL(&ns1, &scale, &a[j * lda], &int1);
            SLC_DSCAL(&np, &scale, &c[j * ldc], &int1);
        }

        *nsys = ns1;
    } else {
        f64 neg_inv_eps = -one / eps;
        SLC_DLASET("F", &n, &n, &zero, &neg_inv_eps, a, &lda);
        SLC_DLASET("F", &n, &m, &zero, &zero, b, &ldb);
        SLC_DLASET("F", &np, &n, &zero, &zero, c, &ldc);
        *nsys = n;
    }

    dwork[0] = (f64)lwamax;
}
