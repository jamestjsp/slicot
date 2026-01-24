// SPDX-License-Identifier: BSD-3-Clause
// Complex pencil balancing - translation of SLICOT MB4DLZ

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <complex.h>
#include <stdbool.h>

// Statement function: CABS1(z) = |Re(z)| + |Im(z)|
static inline f64 cabs1(c128 z) {
    return fabs(creal(z)) + fabs(cimag(z));
}

void mb4dlz(const char *job, i32 n, f64 thresh, c128 *a, i32 lda,
            c128 *b, i32 ldb, i32 *ilo, i32 *ihi,
            f64 *lscale, f64 *rscale, f64 *dwork, i32 *iwarn, i32 *info) {

    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;
    const f64 THREE = 3.0;
    const f64 ZERO = 0.0;
    const f64 MXGAIN = 100.0;
    const f64 SCLFAC = 10.0;
    const c128 CZERO = 0.0 + 0.0*I;

    bool lperm, lscal, evnorm, loop, stormn;
    i32 i, icab, iflow, ip1, ir, irab, it, iter, ith;
    i32 j, jc, jp1, k, kount, ks, kw1, kw2, kw3, kw4, kw5, kw6, kw7;
    i32 l, lm1, lrab, lsfmax, lsfmin, m, nr, nrp2;
    f64 ab, alpha, basl, beta, cab, cmax, coef, coef2, coef5, cor;
    f64 denom, eps, ew, ewc, gamma, gap, minpro, minrat, mn, mx;
    f64 mxcond, mxnorm, mxs, na, na0, nas, nb, nb0, nbs, pgamma;
    f64 prod, rab, ratio, sfmax, sfmin, sum, t, ta, tb, tc, th, th0, ths;
    f64 dum;
    i32 int1 = 1, int0 = 0;

    *info = 0;
    *iwarn = 0;

    lperm = (*job == 'P' || *job == 'p' || *job == 'B' || *job == 'b');
    lscal = (*job == 'S' || *job == 's' || *job == 'B' || *job == 'b');

    if (!lperm && !lscal && !(*job == 'N' || *job == 'n')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 0 ? n : 1)) {
        *info = -5;
    } else if (ldb < (n > 0 ? n : 1)) {
        *info = -7;
    }

    if (*info != 0) {
        return;
    }

    *ilo = 1;
    *ihi = n;

    // Quick return if possible
    if (n == 0) {
        return;
    }

    if ((!lperm && !lscal) || n == 1) {
        dum = ONE;
        SLC_DCOPY(&n, &dum, &int0, lscale, &int1);
        SLC_DCOPY(&n, &dum, &int0, rscale, &int1);
        if (n == 1 && lscal) {
            na0 = cabs(a[0]);
            nb0 = cabs(b[0]);
            dwork[0] = na0;
            dwork[1] = nb0;
            dwork[2] = na0;
            dwork[3] = nb0;
            dwork[4] = thresh;
        }
        return;
    }

    k = 1;
    l = n;

    if (lperm) {
        // Permute the matrices A and B to isolate the eigenvalues.

        // Find row with one nonzero in columns 1 through L.
    label10:
        lm1 = l - 1;
        for (i = l; i >= 1; i--) {
            // Search columns 1 to lm1
            for (j = 1; j <= lm1; j++) {
                jp1 = j + 1;
                // A(i,j) in Fortran = a[(i-1) + (j-1)*lda]
                if (a[(i-1) + (j-1)*lda] != CZERO || b[(i-1) + (j-1)*ldb] != CZERO) {
                    goto label30;
                }
            }
            j = l;
            goto label50;

        label30:
            // Search columns jp1 to l
            for (j = jp1; j <= l; j++) {
                if (a[(i-1) + (j-1)*lda] != CZERO || b[(i-1) + (j-1)*ldb] != CZERO) {
                    goto label60_continue;
                }
            }
            j = jp1 - 1;

        label50:
            m = l;
            iflow = 1;
            goto label130;

        label60_continue:;
        }

        // Find column with one nonzero in rows K through N.
    label70:
        for (j = k; j <= l; j++) {
            // Search rows k to lm1
            for (i = k; i <= lm1; i++) {
                ip1 = i + 1;
                if (a[(i-1) + (j-1)*lda] != CZERO || b[(i-1) + (j-1)*ldb] != CZERO) {
                    goto label90;
                }
            }
            i = l;
            goto label110;

        label90:
            // Search rows ip1 to l
            for (i = ip1; i <= l; i++) {
                if (a[(i-1) + (j-1)*lda] != CZERO || b[(i-1) + (j-1)*ldb] != CZERO) {
                    goto label120_continue;
                }
            }
            i = ip1 - 1;

        label110:
            m = k;
            iflow = 2;
            goto label130;

        label120_continue:;
        }
        goto label140;

    label130:
        // Permute rows M and I
        lscale[m-1] = (f64)i;
        if (i != m) {
            i32 len = n - k + 1;
            SLC_ZSWAP(&len, &a[(i-1) + (k-1)*lda], &lda, &a[(m-1) + (k-1)*lda], &lda);
            SLC_ZSWAP(&len, &b[(i-1) + (k-1)*ldb], &ldb, &b[(m-1) + (k-1)*ldb], &ldb);
        }

        // Permute columns M and J
        rscale[m-1] = (f64)j;
        if (j != m) {
            SLC_ZSWAP(&l, &a[(j-1)*lda], &int1, &a[(m-1)*lda], &int1);
            SLC_ZSWAP(&l, &b[(j-1)*ldb], &int1, &b[(m-1)*ldb], &int1);
        }

        if (iflow == 1) {
            l = lm1;
            if (l != 1) {
                goto label10;
            }
            rscale[0] = ONE;
            lscale[0] = ONE;
        } else {
            k = k + 1;
            goto label70;
        }
    }

label140:
    *ilo = k;
    *ihi = l;

    if (!lscal) {
        for (i = *ilo; i <= *ihi; i++) {
            lscale[i-1] = ONE;
            rscale[i-1] = ONE;
        }
        return;
    }

    nr = *ihi - *ilo + 1;

    // Compute initial 1-norms and return if ILO = N.
    na0 = SLC_ZLANGE("1", &nr, &nr, &a[(*ilo-1) + (*ilo-1)*lda], &lda, dwork);
    nb0 = SLC_ZLANGE("1", &nr, &nr, &b[(*ilo-1) + (*ilo-1)*ldb], &ldb, dwork);

    if (*ilo == *ihi) {
        dwork[0] = na0;
        dwork[1] = nb0;
        dwork[2] = na0;
        dwork[3] = nb0;
        dwork[4] = thresh;
        return;
    }

    // Balance the submatrices in rows ILO to IHI.
    // Initialize balancing and allocate work storage.
    kw1 = n;
    kw2 = kw1 + n;
    kw3 = kw2 + n;
    kw4 = kw3 + n;
    kw5 = kw4 + n;
    dum = ZERO;

    // Prepare for scaling.
    sfmin = SLC_DLAMCH("Safe minimum");
    sfmax = ONE / sfmin;
    basl = log10(SCLFAC);
    lsfmin = (i32)(log10(sfmin) / basl + ONE);
    lsfmax = (i32)(log10(sfmax) / basl);
    mxnorm = (na0 > nb0) ? na0 : nb0;
    loop = thresh < ZERO;

    if (loop) {
        // Compute relative threshold.
        na = na0;
        nas = na0;
        nb = nb0;
        nbs = nb0;

        ith = (i32)thresh;
        mxs = mxnorm;
        mx = ZERO;
        mn = sfmax;

        if (ith >= -2) {
            if (na < nb) {
                ratio = (nb/na < sfmax) ? nb/na : sfmax;
            } else {
                ratio = (na/nb < sfmax) ? na/nb : sfmax;
            }
            minrat = ratio;
        } else if (ith <= -10) {
            mxcond = -thresh;
        } else {
            denom = (ONE > mxnorm) ? ONE : mxnorm;
            prod = (na/denom) * (nb/denom);
            minpro = prod;
        }
        stormn = false;
        evnorm = false;

        // Find maximum order of magnitude of the differences in sizes
        // of the nonzero entries, not considering diag(A) and diag(B).
        for (j = *ilo; j <= *ihi; j++) {
            for (i = *ilo; i <= *ihi; i++) {
                if (i != j) {
                    ab = cabs1(a[(i-1) + (j-1)*lda]);
                    if (ab != ZERO) {
                        mn = (mn < ab) ? mn : ab;
                    }
                    mx = (mx > ab) ? mx : ab;
                }
            }
        }

        for (j = *ilo; j <= *ihi; j++) {
            for (i = *ilo; i <= *ihi; i++) {
                if (i != j) {
                    ab = cabs1(b[(i-1) + (j-1)*ldb]);
                    if (ab != ZERO) {
                        mn = (mn < ab) ? mn : ab;
                    }
                    mx = (mx > ab) ? mx : ab;
                }
            }
        }

        if (mx * sfmin <= mn) {
            gap = mx / mn;
        } else {
            gap = sfmax;
        }

        eps = SLC_DLAMCH("Precision");
        {
            i32 tmp1 = (i32)log10(gap);
            i32 tmp2 = -(i32)log10(eps);
            iter = (tmp1 < tmp2 ? tmp1 : tmp2) + 1;
        }
        th = ((mn > mx*eps) ? mn : mx*eps) / ((mxnorm > sfmin) ? mxnorm : sfmin);
        ths = th;
        kw6 = kw5 + n + *ilo;
        kw7 = kw6 + n;
        SLC_DCOPY(&nr, &lscale[*ilo-1], &int1, &dwork[kw6], &int1);
        SLC_DCOPY(&nr, &rscale[*ilo-1], &int1, &dwork[kw7], &int1);

        // Set the maximum condition number of the transformations.
        if (ith > -10) {
            mxcond = ONE / sqrt(eps);
        }
    } else {
        th = mxnorm * thresh;
        iter = 1;
        evnorm = true;
    }
    th0 = th;

    coef = ONE / (f64)(2*nr);
    coef2 = coef * coef;
    coef5 = HALF * coef2;
    nrp2 = nr + 2;
    beta = ZERO;

    // If THRESH < 0, use a loop to reduce the norm ratio.
    for (i32 kiter = 1; kiter <= iter; kiter++) {
        // Compute right side vector in resulting linear equations.
        i32 six_n = 6 * n;
        dum = ZERO;
        SLC_DCOPY(&six_n, &dum, &int0, dwork, &int1);
        SLC_DCOPY(&nr, &dum, &int0, &lscale[*ilo-1], &int1);
        SLC_DCOPY(&nr, &dum, &int0, &rscale[*ilo-1], &int1);

        for (i = *ilo; i <= *ihi; i++) {
            for (j = *ilo; j <= *ihi; j++) {
                ta = cabs1(a[(i-1) + (j-1)*lda]);
                tb = cabs1(b[(i-1) + (j-1)*ldb]);
                if (ta > th) {
                    ta = log10(ta) / basl;
                } else {
                    ta = ZERO;
                }
                if (tb > th) {
                    tb = log10(tb) / basl;
                } else {
                    tb = ZERO;
                }
                dwork[(i-1)+kw4] = dwork[(i-1)+kw4] - ta - tb;
                dwork[(j-1)+kw5] = dwork[(j-1)+kw5] - ta - tb;
            }
        }

        it = 1;

        // Start generalized conjugate gradient iteration.
    label220:
        gamma = SLC_DDOT(&nr, &dwork[*ilo-1+kw4], &int1, &dwork[*ilo-1+kw4], &int1) +
                SLC_DDOT(&nr, &dwork[*ilo-1+kw5], &int1, &dwork[*ilo-1+kw5], &int1);

        ew = ZERO;
        ewc = ZERO;
        for (i = *ilo; i <= *ihi; i++) {
            ew = ew + dwork[(i-1)+kw4];
            ewc = ewc + dwork[(i-1)+kw5];
        }

        gamma = coef * gamma - coef2 * (ew*ew + ewc*ewc) - coef5 * (ew - ewc) * (ew - ewc);
        if (gamma == ZERO) {
            goto label300;
        }
        if (it != 1) {
            beta = gamma / pgamma;
        }
        t = coef5 * (ewc - THREE * ew);
        tc = coef5 * (ew - THREE * ewc);

        SLC_DSCAL(&nr, &beta, &dwork[*ilo-1], &int1);
        SLC_DSCAL(&nr, &beta, &dwork[*ilo-1+kw1], &int1);

        SLC_DAXPY(&nr, &coef, &dwork[*ilo-1+kw4], &int1, &dwork[*ilo-1+kw1], &int1);
        SLC_DAXPY(&nr, &coef, &dwork[*ilo-1+kw5], &int1, &dwork[*ilo-1], &int1);

        for (j = *ilo; j <= *ihi; j++) {
            dwork[(j-1)] = dwork[(j-1)] + tc;
            dwork[(j-1)+kw1] = dwork[(j-1)+kw1] + t;
        }

        // Apply matrix to vector.
        for (i = *ilo; i <= *ihi; i++) {
            kount = 0;
            sum = ZERO;
            for (j = *ilo; j <= *ihi; j++) {
                ks = kount;
                if (a[(i-1) + (j-1)*lda] != CZERO) {
                    kount = kount + 1;
                }
                if (b[(i-1) + (j-1)*ldb] != CZERO) {
                    kount = kount + 1;
                }
                sum = sum + (f64)(kount - ks) * dwork[(j-1)];
            }
            dwork[(i-1)+kw2] = (f64)kount * dwork[(i-1)+kw1] + sum;
        }

        for (j = *ilo; j <= *ihi; j++) {
            kount = 0;
            sum = ZERO;
            for (i = *ilo; i <= *ihi; i++) {
                ks = kount;
                if (a[(i-1) + (j-1)*lda] != CZERO) {
                    kount = kount + 1;
                }
                if (b[(i-1) + (j-1)*ldb] != CZERO) {
                    kount = kount + 1;
                }
                sum = sum + (f64)(kount - ks) * dwork[(i-1)+kw1];
            }
            dwork[(j-1)+kw3] = (f64)kount * dwork[(j-1)] + sum;
        }

        sum = SLC_DDOT(&nr, &dwork[*ilo-1+kw1], &int1, &dwork[*ilo-1+kw2], &int1) +
              SLC_DDOT(&nr, &dwork[*ilo-1], &int1, &dwork[*ilo-1+kw3], &int1);
        alpha = gamma / sum;

        // Determine correction to current iteration.
        cmax = ZERO;
        for (i = *ilo; i <= *ihi; i++) {
            cor = alpha * dwork[(i-1)+kw1];
            if (fabs(cor) > cmax) {
                cmax = fabs(cor);
            }
            lscale[i-1] = lscale[i-1] + cor;
            cor = alpha * dwork[(i-1)];
            if (fabs(cor) > cmax) {
                cmax = fabs(cor);
            }
            rscale[i-1] = rscale[i-1] + cor;
        }

        if (cmax >= HALF) {
            f64 neg_alpha = -alpha;
            SLC_DAXPY(&nr, &neg_alpha, &dwork[*ilo-1+kw2], &int1, &dwork[*ilo-1+kw4], &int1);
            SLC_DAXPY(&nr, &neg_alpha, &dwork[*ilo-1+kw3], &int1, &dwork[*ilo-1+kw5], &int1);

            pgamma = gamma;
            it = it + 1;
            if (it <= nrp2) {
                goto label220;
            }
        }

        // End generalized conjugate gradient iteration.
    label300:
        // Compute diagonal scaling matrices.
        for (i = *ilo; i <= *ihi; i++) {
            i32 len = n - *ilo + 1;
            irab = SLC_IZAMAX(&len, &a[(i-1) + (*ilo-1)*lda], &lda);
            rab = cabs(a[(i-1) + (*ilo-1+irab-1)*lda]);
            irab = SLC_IZAMAX(&len, &b[(i-1) + (*ilo-1)*ldb], &ldb);
            rab = (rab > cabs(b[(i-1) + (*ilo-1+irab-1)*ldb])) ? rab : cabs(b[(i-1) + (*ilo-1+irab-1)*ldb]);
            lrab = (i32)(log10(rab + sfmin) / basl + ONE);
            ir = (i32)(lscale[i-1] + copysign(HALF, lscale[i-1]));
            ir = (ir < lsfmin) ? lsfmin : ir;
            ir = (ir > lsfmax) ? lsfmax : ir;
            ir = (ir > lsfmax - lrab) ? lsfmax - lrab : ir;
            lscale[i-1] = pow(SCLFAC, ir);

            icab = SLC_IZAMAX(ihi, &a[(i-1)*lda], &int1);
            cab = cabs(a[(icab-1) + (i-1)*lda]);
            icab = SLC_IZAMAX(ihi, &b[(i-1)*ldb], &int1);
            cab = (cab > cabs(b[(icab-1) + (i-1)*ldb])) ? cab : cabs(b[(icab-1) + (i-1)*ldb]);
            lrab = (i32)(log10(cab + sfmin) / basl + ONE);
            jc = (i32)(rscale[i-1] + copysign(HALF, rscale[i-1]));
            jc = (jc < lsfmin) ? lsfmin : jc;
            jc = (jc > lsfmax) ? lsfmax : jc;
            jc = (jc > lsfmax - lrab) ? lsfmax - lrab : jc;
            rscale[i-1] = pow(SCLFAC, jc);
        }

        // Check if all scaling factors equal 1.
        bool all_ones = true;
        for (i = *ilo; i <= *ihi; i++) {
            if (lscale[i-1] != ONE || rscale[i-1] != ONE) {
                all_ones = false;
                break;
            }
        }

        if (all_ones) {
            // Finish the procedure for all scaling factors equal to 1.
            nas = na0;
            nbs = nb0;
            ths = th0;
            goto label460;
        }

        // label330
        if (loop) {
            if (ith <= -10) {
                // Compute the reciprocal condition number of the left and
                // right transformations. Continue the loop if it is too small.
                ir = SLC_IDAMAX(&nr, &lscale[*ilo-1], &int1);
                jc = SLC_IDAMAX(&nr, &rscale[*ilo-1], &int1);
                t = lscale[*ilo-1+ir-1];
                mn = t;
                for (i = *ilo; i <= *ihi; i++) {
                    if (lscale[i-1] < mn) {
                        mn = lscale[i-1];
                    }
                }
                t = mn / t;
                ta = rscale[*ilo-1+jc-1];
                mn = ta;
                for (i = *ilo; i <= *ihi; i++) {
                    if (rscale[i-1] < mn) {
                        mn = rscale[i-1];
                    }
                }
                t = (t < mn/ta) ? t : mn/ta;
                if (t < ONE/mxcond) {
                    th = th * TEN;
                    continue;  // GO TO 400
                } else {
                    ths = th;
                    evnorm = true;
                    goto label430;
                }
            }

            // Compute the 1-norms of the scaled submatrices,
            // without actually scaling them.
            na = ZERO;
            for (j = *ilo; j <= *ihi; j++) {
                t = ZERO;
                for (i = *ilo; i <= *ihi; i++) {
                    t = t + cabs(a[(i-1) + (j-1)*lda]) * lscale[i-1] * rscale[j-1];
                }
                if (t > na) {
                    na = t;
                }
            }

            nb = ZERO;
            for (j = *ilo; j <= *ihi; j++) {
                t = ZERO;
                for (i = *ilo; i <= *ihi; i++) {
                    t = t + cabs(b[(i-1) + (j-1)*ldb]) * lscale[i-1] * rscale[j-1];
                }
                if (t > nb) {
                    nb = t;
                }
            }

            if (ith >= -4 && ith < -2) {
                prod = (na/denom) * (nb/denom);
                if (minpro > prod) {
                    minpro = prod;
                    stormn = true;
                    SLC_DCOPY(&nr, &lscale[*ilo-1], &int1, &dwork[kw6], &int1);
                    SLC_DCOPY(&nr, &rscale[*ilo-1], &int1, &dwork[kw7], &int1);
                    nas = na;
                    nbs = nb;
                    ths = th;
                }
            } else if (ith >= -2) {
                if (na < nb) {
                    ratio = (nb/na < sfmax) ? nb/na : sfmax;
                } else {
                    ratio = (na/nb < sfmax) ? na/nb : sfmax;
                }
                if (minrat > ratio) {
                    minrat = ratio;
                    stormn = true;
                    SLC_DCOPY(&nr, &lscale[*ilo-1], &int1, &dwork[kw6], &int1);
                    SLC_DCOPY(&nr, &rscale[*ilo-1], &int1, &dwork[kw7], &int1);
                    mxs = (na > nb) ? na : nb;
                    nas = na;
                    nbs = nb;
                    ths = th;
                }
            }
            th = th * TEN;
        }
    }  // end for kiter

    // Prepare for scaling.
    if (loop) {
        if (ith <= -10) {
            // Could not find enough well conditioned transformations
            // for THRESH <= -10. Set scaling factors to 1 and return.
            dum = ONE;
            SLC_DCOPY(&nr, &dum, &int0, &lscale[*ilo-1], &int1);
            SLC_DCOPY(&nr, &dum, &int0, &rscale[*ilo-1], &int1);
            *iwarn = 1;
            goto label460;
        }

        // Check if scaling might reduce the accuracy when solving related
        // eigenproblems, and set the scaling factors to 1 in this case,
        // if THRESH = -2 or THRESH = -4.
        if ((mxnorm < mxs && mxnorm < mxs/MXGAIN && ith == -2) || ith == -4) {
            ir = SLC_IDAMAX(&nr, &dwork[kw6], &int1);
            jc = SLC_IDAMAX(&nr, &dwork[kw7], &int1);
            t = dwork[kw6+ir-1];
            mn = t;
            for (i = kw6; i < kw6+nr; i++) {
                if (dwork[i] < mn) {
                    mn = dwork[i];
                }
            }
            t = mn / t;
            ta = dwork[kw7+jc-1];
            mn = ta;
            for (i = kw7; i < kw7+nr; i++) {
                if (dwork[i] < mn) {
                    mn = dwork[i];
                }
            }
            t = (t < mn/ta) ? t : mn/ta;
            if (t < ONE/mxcond) {
                dum = ONE;
                SLC_DCOPY(&nr, &dum, &int0, &lscale[*ilo-1], &int1);
                SLC_DCOPY(&nr, &dum, &int0, &rscale[*ilo-1], &int1);
                *iwarn = 1;
                nas = na0;
                nbs = nb0;
                ths = th0;
                goto label460;
            }
        }
        if (stormn) {
            SLC_DCOPY(&nr, &dwork[kw6], &int1, &lscale[*ilo-1], &int1);
            SLC_DCOPY(&nr, &dwork[kw7], &int1, &rscale[*ilo-1], &int1);
        } else {
            nas = na;
            nbs = nb;
            ths = th;
        }
    }

label430:
    // Row scaling.
    for (i = *ilo; i <= *ihi; i++) {
        i32 len = n - *ilo + 1;
        SLC_ZDSCAL(&len, &lscale[i-1], &a[(i-1) + (*ilo-1)*lda], &lda);
        SLC_ZDSCAL(&len, &lscale[i-1], &b[(i-1) + (*ilo-1)*ldb], &ldb);
    }

    // Column scaling.
    for (j = *ilo; j <= *ihi; j++) {
        SLC_ZDSCAL(ihi, &rscale[j-1], &a[(j-1)*lda], &int1);
        SLC_ZDSCAL(ihi, &rscale[j-1], &b[(j-1)*ldb], &int1);
    }

    // Set DWORK(1:5).
label460:
    if (evnorm) {
        nas = SLC_ZLANGE("1", &nr, &nr, &a[(*ilo-1) + (*ilo-1)*lda], &lda, dwork);
        nbs = SLC_ZLANGE("1", &nr, &nr, &b[(*ilo-1) + (*ilo-1)*ldb], &ldb, dwork);
    }

    dwork[0] = na0;
    dwork[1] = nb0;
    dwork[2] = nas;
    dwork[3] = nbs;
    if (loop) {
        dwork[4] = ths / ((mxnorm > sfmin) ? mxnorm : sfmin);
    } else {
        dwork[4] = thresh;
    }

    return;
}
