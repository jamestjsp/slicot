/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * MB4DPZ - Balance a complex skew-Hamiltonian/Hamiltonian pencil
 *
 * Purpose:
 *   To balance the 2*N-by-2*N complex skew-Hamiltonian/Hamiltonian
 *   pencil aS - bH, with
 *
 *         (  A  D  )         (  C  V  )
 *     S = (        ) and H = (        ),  A, C N-by-N,
 *         (  E  A' )         (  W -C' )
 *
 *   where D and E are skew-Hermitian, V and W are Hermitian matrices,
 *   and ' denotes conjugate transpose.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <ctype.h>

void mb4dpz(const char *job, i32 n, f64 thresh, c128 *a, i32 lda,
            c128 *de, i32 ldde, c128 *c, i32 ldc, c128 *vw, i32 ldvw,
            i32 *ilo, f64 *lscale, f64 *rscale, f64 *dwork,
            i32 *iwarn, i32 *info)
{
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;
    const f64 TWO = 2.0;
    const f64 ZERO = 0.0;
    const f64 MXGAIN = 100.0;
    const f64 SCLFAC = 10.0;
    const c128 CZERO = 0.0 + 0.0*I;

    bool lperm, lscal, evnorm, loop, stormn;
    i32 i, icab, iloold, ir, irab, it, iter, ith, j, jc, k, kount, ks;
    i32 kw1, kw2, kw3, kw4, kw5, kw6, kw7, lrab, lsfmax, lsfmin, nr, nrp2;
    f64 ab, alpha, basl, beta, cab, cmax, coef, coef2, coef5, cor;
    f64 denom, eps, ew, gamma, gap, minpro, minrat, mn, mx;
    f64 mxcond, mxnorm, mxs, nh, nh0, nhs, ns, ns0, nss, pgamma;
    f64 prod, rab, ratio, sfmax, sfmin, sum, t, ta, tc, td, te, th, th0;
    f64 ths, tv, tw;
    f64 dum;
    i32 int1 = 1, int0 = 0;

    *info = 0;
    *iwarn = 0;

    char job_upper = toupper((unsigned char)job[0]);
    lperm = (job_upper == 'P' || job_upper == 'B');
    lscal = (job_upper == 'S' || job_upper == 'B');

    if (!lperm && !lscal && job_upper != 'N') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 0 ? n : 1)) {
        *info = -5;
    } else if (ldde < (n > 0 ? n : 1)) {
        *info = -7;
    } else if (ldc < (n > 0 ? n : 1)) {
        *info = -9;
    } else if (ldvw < (n > 0 ? n : 1)) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    *ilo = 1;

    // Quick return if possible
    if (n == 0) {
        return;
    }

    if ((!lperm && !lscal) || n == 1) {
        dum = ONE;
        SLC_DCOPY(&n, &dum, &int0, lscale, &int1);
        SLC_DCOPY(&n, &dum, &int0, rscale, &int1);
        if (n == 1 && lscal) {
            ns0 = ma02iz("skew-Hamiltonian", "1-norm", 1, a, lda, de, ldde, dwork);
            nh0 = ma02iz("Hamiltonian", "1-norm", 1, c, ldc, vw, ldvw, dwork);
            dwork[0] = ns0;
            dwork[1] = nh0;
            dwork[2] = ns0;
            dwork[3] = nh0;
            dwork[4] = thresh;
        }
        return;
    }

    if (lperm) {
        // Permute the matrices S and H to isolate the eigenvalues.
        iloold = 0;

        // WHILE (ILO.NE.ILOOLD)
    label10:
        if (*ilo != iloold) {
            iloold = *ilo;

            // Scan columns ILO .. N
            i = *ilo;
        label20:
            if (i <= n && *ilo == iloold) {
                // Check if column i can be isolated
                // Scan A(ILO:I-1, I), C(ILO:I-1, I)
                for (j = *ilo; j <= i - 1; j++) {
                    if (a[(j-1) + (i-1)*lda] != CZERO ||
                        c[(j-1) + (i-1)*ldc] != CZERO) {
                        i++;
                        goto label20;
                    }
                }
                // Scan A(I+1:N, I), C(I+1:N, I)
                for (j = i + 1; j <= n; j++) {
                    if (a[(j-1) + (i-1)*lda] != CZERO ||
                        c[(j-1) + (i-1)*ldc] != CZERO) {
                        i++;
                        goto label20;
                    }
                }
                // Scan DE(I, ILO:I-1), VW(I, ILO:I-1)
                for (j = *ilo; j <= i - 1; j++) {
                    if (de[(i-1) + (j-1)*ldde] != CZERO ||
                        vw[(i-1) + (j-1)*ldvw] != CZERO) {
                        i++;
                        goto label20;
                    }
                }
                // Check diagonal: DE(I,I) imag part, VW(I,I) real part
                if (cimag(de[(i-1) + (i-1)*ldde]) != ZERO) {
                    i++;
                    goto label20;
                }
                if (creal(vw[(i-1) + (i-1)*ldvw]) != ZERO) {
                    i++;
                    goto label20;
                }
                // Scan DE(I+1:N, I), VW(I+1:N, I)
                for (j = i + 1; j <= n; j++) {
                    if (de[(j-1) + (i-1)*ldde] != CZERO ||
                        vw[(j-1) + (i-1)*ldvw] != CZERO) {
                        i++;
                        goto label20;
                    }
                }

                // Exchange columns/rows ILO <-> I
                lscale[*ilo - 1] = (f64)i;
                rscale[*ilo - 1] = (f64)i;

                if (*ilo != i) {
                    // Swap columns ILO and I in A
                    SLC_ZSWAP(&n, &a[((*ilo)-1)*lda], &int1, &a[(i-1)*lda], &int1);
                    // Swap rows ILO and I in A (starting from ILO)
                    i32 len = n - *ilo + 1;
                    SLC_ZSWAP(&len, &a[(*ilo-1) + (*ilo-1)*lda], &lda,
                              &a[(i-1) + (*ilo-1)*lda], &lda);

                    // Swap in DE using ma02nz
                    ma02nz("Lower", "No transpose", "Skew", n, *ilo, i, de, ldde);
                    ma02nz("Upper", "No transpose", "Skew", n, *ilo, i, &de[ldde], ldde);

                    // Swap columns ILO and I in C
                    SLC_ZSWAP(&n, &c[((*ilo)-1)*ldc], &int1, &c[(i-1)*ldc], &int1);
                    // Swap rows ILO and I in C
                    SLC_ZSWAP(&len, &c[(*ilo-1) + (*ilo-1)*ldc], &ldc,
                              &c[(i-1) + (*ilo-1)*ldc], &ldc);

                    // Swap in VW using ma02nz
                    ma02nz("Lower", "No transpose", "Not Skew", n, *ilo, i, vw, ldvw);
                    ma02nz("Upper", "No transpose", "Not Skew", n, *ilo, i, &vw[ldvw], ldvw);
                }
                (*ilo)++;
            }
            // END WHILE 20

            // Scan columns N+ILO .. 2*N
            i = *ilo;
        label70:
            if (i <= n && *ilo == iloold) {
                // Check row i of A and C
                for (j = *ilo; j <= i - 1; j++) {
                    if (a[(i-1) + (j-1)*lda] != CZERO ||
                        c[(i-1) + (j-1)*ldc] != CZERO) {
                        i++;
                        goto label70;
                    }
                }
                for (j = i + 1; j <= n; j++) {
                    if (a[(i-1) + (j-1)*lda] != CZERO ||
                        c[(i-1) + (j-1)*ldc] != CZERO) {
                        i++;
                        goto label70;
                    }
                }
                // Check DE(ILO:I-1, I+1), VW(ILO:I-1, I+1)
                for (j = *ilo; j <= i - 1; j++) {
                    if (de[(j-1) + i*ldde] != CZERO ||
                        vw[(j-1) + i*ldvw] != CZERO) {
                        i++;
                        goto label70;
                    }
                }
                // Check diagonal DE(I,I+1), VW(I,I+1)
                if (cimag(de[(i-1) + i*ldde]) != ZERO) {
                    i++;
                    goto label70;
                }
                if (creal(vw[(i-1) + i*ldvw]) != ZERO) {
                    i++;
                    goto label70;
                }
                // Check DE(I, J+1) and VW(I, J+1) for J = I+1 to N
                for (j = i + 1; j <= n; j++) {
                    if (de[(i-1) + j*ldde] != CZERO ||
                        vw[(i-1) + j*ldvw] != CZERO) {
                        i++;
                        goto label70;
                    }
                }

                // Exchange columns/rows I <-> I+N with symplectic permutation
                lscale[*ilo - 1] = (f64)(n + i);
                rscale[*ilo - 1] = (f64)(n + i);

                // Perform the symplectic exchange
                i32 len_ilo = i - *ilo;
                if (len_ilo > 0) {
                    SLC_ZSWAP(&len_ilo, &a[(i-1) + (*ilo-1)*lda], &lda,
                              &de[(i-1) + (*ilo-1)*ldde], &ldde);
                    f64 neg_one = -ONE;
                    SLC_ZDSCAL(&len_ilo, &neg_one, &a[(i-1) + (*ilo-1)*lda], &lda);
                }

                if (n > i) {
                    i32 len_ni = n - i;
                    SLC_ZSWAP(&len_ni, &a[(i-1) + i*lda], &lda, &de[i + (i-1)*ldde], &int1);
                    for (j = i + 1; j <= n; j++) {
                        a[(i-1) + (j-1)*lda] = conj(a[(i-1) + (j-1)*lda]);
                        de[(j-1) + (i-1)*ldde] = -creal(de[(j-1) + (i-1)*ldde])
                                                 + cimag(de[(j-1) + (i-1)*ldde])*I;
                    }
                }

                i32 len_im1 = i - 1;
                if (len_im1 > 0) {
                    SLC_ZSWAP(&len_im1, &a[(i-1)*lda], &int1, &de[i*ldde], &int1);
                    f64 neg_one = -ONE;
                    SLC_ZDSCAL(&len_im1, &neg_one, &a[(i-1)*lda], &int1);
                }

                if (n > i) {
                    i32 len_ni = n - i;
                    SLC_ZSWAP(&len_ni, &a[i + (i-1)*lda], &int1,
                              &de[(i-1) + (i+1)*ldde], &ldde);
                    for (j = i + 1; j <= n; j++) {
                        a[(j-1) + (i-1)*lda] = conj(a[(j-1) + (i-1)*lda]);
                        de[(i-1) + j*ldde] = -creal(de[(i-1) + j*ldde])
                                             + cimag(de[(i-1) + j*ldde])*I;
                    }
                }
                a[(i-1) + (i-1)*lda] = conj(a[(i-1) + (i-1)*lda]);
                c128 t_val = de[(i-1) + (i-1)*ldde];
                de[(i-1) + (i-1)*ldde] = -de[(i-1) + i*ldde];
                de[(i-1) + i*ldde] = -t_val;

                // Same for C and VW
                if (len_ilo > 0) {
                    SLC_ZSWAP(&len_ilo, &c[(i-1) + (*ilo-1)*ldc], &ldc,
                              &vw[(i-1) + (*ilo-1)*ldvw], &ldvw);
                    f64 neg_one = -ONE;
                    SLC_ZDSCAL(&len_ilo, &neg_one, &c[(i-1) + (*ilo-1)*ldc], &ldc);
                }

                if (n > i) {
                    i32 len_ni = n - i;
                    SLC_ZSWAP(&len_ni, &c[(i-1) + i*ldc], &ldc, &vw[i + (i-1)*ldvw], &int1);
                    for (j = i + 1; j <= n; j++) {
                        vw[(j-1) + (i-1)*ldvw] = conj(vw[(j-1) + (i-1)*ldvw]);
                        c[(i-1) + (j-1)*ldc] = -creal(c[(i-1) + (j-1)*ldc])
                                               + cimag(c[(i-1) + (j-1)*ldc])*I;
                    }
                }

                if (len_im1 > 0) {
                    SLC_ZSWAP(&len_im1, &c[(i-1)*ldc], &int1, &vw[i*ldvw], &int1);
                    f64 neg_one = -ONE;
                    SLC_ZDSCAL(&len_im1, &neg_one, &c[(i-1)*ldc], &int1);
                }

                if (n > i) {
                    i32 len_ni = n - i;
                    SLC_ZSWAP(&len_ni, &c[i + (i-1)*ldc], &int1,
                              &vw[(i-1) + (i+1)*ldvw], &ldvw);
                    for (j = i + 1; j <= n; j++) {
                        vw[(i-1) + j*ldvw] = conj(vw[(i-1) + j*ldvw]);
                        c[(j-1) + (i-1)*ldc] = -creal(c[(j-1) + (i-1)*ldc])
                                               + cimag(c[(j-1) + (i-1)*ldc])*I;
                    }
                }
                c[(i-1) + (i-1)*ldc] = -conj(c[(i-1) + (i-1)*ldc]);
                t_val = vw[(i-1) + (i-1)*ldvw];
                vw[(i-1) + (i-1)*ldvw] = -vw[(i-1) + i*ldvw];
                vw[(i-1) + i*ldvw] = -t_val;

                // Exchange columns/rows ILO <-> I
                if (*ilo != i) {
                    SLC_ZSWAP(&n, &a[((*ilo)-1)*lda], &int1, &a[(i-1)*lda], &int1);
                    i32 len = n - *ilo + 1;
                    SLC_ZSWAP(&len, &a[(*ilo-1) + (*ilo-1)*lda], &lda,
                              &a[(i-1) + (*ilo-1)*lda], &lda);

                    ma02nz("Lower", "No transpose", "Skew", n, *ilo, i, de, ldde);
                    ma02nz("Upper", "No transpose", "Skew", n, *ilo, i, &de[ldde], ldde);

                    SLC_ZSWAP(&n, &c[((*ilo)-1)*ldc], &int1, &c[(i-1)*ldc], &int1);
                    SLC_ZSWAP(&len, &c[(*ilo-1) + (*ilo-1)*ldc], &ldc,
                              &c[(i-1) + (*ilo-1)*ldc], &ldc);

                    ma02nz("Lower", "No transpose", "Not Skew", n, *ilo, i, vw, ldvw);
                    ma02nz("Upper", "No transpose", "Not Skew", n, *ilo, i, &vw[ldvw], ldvw);
                }
                (*ilo)++;
            }
            // END WHILE 70
            goto label10;
        }
        // END WHILE 10

        for (i = *ilo; i <= n; i++) {
            lscale[i - 1] = ONE;
            rscale[i - 1] = ONE;
        }
        if (!lscal) {
            return;
        }
    }

    nr = n - *ilo + 1;

    // Compute initial 1-norms and return if ILO = N
    ns0 = ma02iz("skew-Hamiltonian", "1-norm", nr, &a[(*ilo-1) + (*ilo-1)*lda], lda,
                 &de[(*ilo-1) + (*ilo-1)*ldde], ldde, dwork);
    nh0 = ma02iz("Hamiltonian", "1-norm", nr, &c[(*ilo-1) + (*ilo-1)*ldc], ldc,
                 &vw[(*ilo-1) + (*ilo-1)*ldvw], ldvw, dwork);

    if (*ilo == n) {
        dwork[0] = ns0;
        dwork[1] = nh0;
        dwork[2] = ns0;
        dwork[3] = nh0;
        dwork[4] = thresh;
        return;
    }

    // Balance the submatrices in rows ILO to N
    // Initialize balancing and allocate work storage
    kw1 = n;
    kw2 = kw1 + n;
    kw3 = kw2 + n;
    kw4 = kw3 + n;
    kw5 = kw4 + n;
    dum = ZERO;

    // Prepare for scaling
    sfmin = SLC_DLAMCH("Safe minimum");
    sfmax = ONE / sfmin;
    basl = log10(SCLFAC);
    lsfmin = (i32)(log10(sfmin) / basl + ONE);
    lsfmax = (i32)(log10(sfmax) / basl);
    mxnorm = (ns0 > nh0) ? ns0 : nh0;
    loop = thresh < ZERO;

    if (loop) {
        // Compute relative threshold
        ns = ns0;
        nss = ns0;
        nh = nh0;
        nhs = nh0;

        ith = (i32)thresh;
        mxs = mxnorm;
        mx = ZERO;
        mn = sfmax;

        if (ith >= -2) {
            if (ns < nh) {
                ratio = (nh/ns < sfmax) ? nh/ns : sfmax;
            } else {
                ratio = (ns/nh < sfmax) ? ns/nh : sfmax;
            }
            minrat = ratio;
        } else if (ith <= -10) {
            mxcond = -thresh;
        } else {
            denom = (ONE > mxnorm) ? ONE : mxnorm;
            prod = (ns/denom) * (nh/denom);
            minpro = prod;
        }
        stormn = false;
        evnorm = false;

        // Find max order of magnitude differences in nonzero entries
        for (j = *ilo; j <= n; j++) {
            for (i = *ilo; i <= n; i++) {
                if (i != j) {
                    ab = cabs(a[(i-1) + (j-1)*lda]);
                    if (ab != ZERO) {
                        mn = (mn < ab) ? mn : ab;
                    }
                    mx = (mx > ab) ? mx : ab;
                }
            }
        }

        for (j = *ilo; j <= n; j++) {
            for (i = *ilo; i <= n; i++) {
                ab = cabs(de[(i-1) + (j-1)*ldde]);
                if (ab != ZERO) {
                    mn = (mn < ab) ? mn : ab;
                }
                mx = (mx > ab) ? mx : ab;
                ab = cabs(de[(i-1) + j*ldde]);
                if (ab != ZERO) {
                    mn = (mn < ab) ? mn : ab;
                }
                mx = (mx > ab) ? mx : ab;
            }
        }

        for (j = *ilo; j <= n; j++) {
            for (i = *ilo; i <= n; i++) {
                if (i != j) {
                    ab = cabs(c[(i-1) + (j-1)*ldc]);
                    if (ab != ZERO) {
                        mn = (mn < ab) ? mn : ab;
                    }
                    mx = (mx > ab) ? mx : ab;
                }
            }
        }

        for (j = *ilo; j <= n; j++) {
            for (i = *ilo; i <= n; i++) {
                ab = cabs(vw[(i-1) + (j-1)*ldvw]);
                if (ab != ZERO) {
                    mn = (mn < ab) ? mn : ab;
                }
                mx = (mx > ab) ? mx : ab;
                ab = cabs(vw[(i-1) + j*ldvw]);
                if (ab != ZERO) {
                    mn = (mn < ab) ? mn : ab;
                }
                mx = (mx > ab) ? mx : ab;
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

        // Set max condition number of transformations
        if (ith > -10) {
            mxcond = ONE / sqrt(eps);
        }
    } else {
        th = mxnorm * thresh;
        iter = 1;
        evnorm = true;
    }
    th0 = th;

    coef = HALF / (f64)(2*nr);
    coef2 = coef * coef;
    coef5 = HALF * coef2;
    nrp2 = nr + 2;
    beta = ZERO;

    // If THRESH < 0, use a loop to reduce the norm ratio
    for (k = 1; k <= iter; k++) {
        // Compute right side vector in resulting linear equations
        i32 six_n = 6 * n;
        dum = ZERO;
        SLC_DCOPY(&six_n, &dum, &int0, dwork, &int1);
        SLC_DCOPY(&nr, &dum, &int0, &lscale[*ilo-1], &int1);
        SLC_DCOPY(&nr, &dum, &int0, &rscale[*ilo-1], &int1);

        for (i = *ilo; i <= n; i++) {
            for (j = *ilo; j <= n; j++) {
                ta = cabs(a[(i-1) + (j-1)*lda]);
                tc = cabs(c[(i-1) + (j-1)*ldc]);
                if (j > i) {
                    td = cabs(de[(i-1) + j*ldde]);
                    te = cabs(de[(j-1) + (i-1)*ldde]);
                    tv = cabs(vw[(i-1) + j*ldvw]);
                    tw = cabs(vw[(j-1) + (i-1)*ldvw]);
                } else {
                    td = cabs(de[(j-1) + i*ldde]);
                    te = cabs(de[(i-1) + (j-1)*ldde]);
                    tv = cabs(vw[(j-1) + i*ldvw]);
                    tw = cabs(vw[(i-1) + (j-1)*ldvw]);
                }
                if (ta > th) {
                    ta = log10(ta) / basl;
                } else {
                    ta = ZERO;
                }
                if (tc > th) {
                    tc = log10(tc) / basl;
                } else {
                    tc = ZERO;
                }
                if (td > th) {
                    td = log10(td) / basl;
                } else {
                    td = ZERO;
                }
                if (te > th) {
                    te = log10(te) / basl;
                } else {
                    te = ZERO;
                }
                if (tv > th) {
                    tv = log10(tv) / basl;
                } else {
                    tv = ZERO;
                }
                if (tw > th) {
                    tw = log10(tw) / basl;
                } else {
                    tw = ZERO;
                }
                dwork[(i-1)+kw4] = dwork[(i-1)+kw4] - ta - tc - td - tv;
                dwork[(j-1)+kw5] = dwork[(j-1)+kw5] - ta - tc - te - tw;
            }
        }

        it = 1;

        // Start generalized conjugate gradient iteration
    label270:
        gamma = SLC_DDOT(&nr, &dwork[*ilo-1+kw4], &int1, &dwork[*ilo-1+kw4], &int1) +
                SLC_DDOT(&nr, &dwork[*ilo-1+kw5], &int1, &dwork[*ilo-1+kw5], &int1);
        gamma *= TWO;

        ew = ZERO;
        for (i = *ilo; i <= n; i++) {
            ew = ew + dwork[(i-1)+kw4] + dwork[(i-1)+kw5];
        }

        gamma = coef*gamma - TWO*coef2*ew*ew;
        if (gamma == ZERO) {
            goto label350;
        }
        if (it != 1) {
            beta = gamma / pgamma;
        }
        t = -TWO*coef5*ew;

        SLC_DSCAL(&nr, &beta, &dwork[*ilo-1], &int1);
        SLC_DSCAL(&nr, &beta, &dwork[*ilo-1+kw1], &int1);

        SLC_DAXPY(&nr, &coef, &dwork[*ilo-1+kw4], &int1, &dwork[*ilo-1+kw1], &int1);
        SLC_DAXPY(&nr, &coef, &dwork[*ilo-1+kw5], &int1, &dwork[*ilo-1], &int1);

        for (j = *ilo; j <= n; j++) {
            dwork[(j-1)] = dwork[(j-1)] + t;
            dwork[(j-1)+kw1] = dwork[(j-1)+kw1] + t;
        }

        // Apply matrix to vector
        for (i = *ilo; i <= n; i++) {
            kount = 0;
            sum = ZERO;
            for (j = *ilo; j <= n; j++) {
                ks = kount;
                if (a[(i-1) + (j-1)*lda] != CZERO) {
                    kount++;
                }
                if (c[(i-1) + (j-1)*ldc] != CZERO) {
                    kount++;
                }
                sum += (f64)(kount - ks) * dwork[(j-1)];

                ks = kount;
                if (j >= i) {
                    if (de[(i-1) + j*ldde] != CZERO) {
                        kount++;
                    }
                    if (vw[(i-1) + j*ldvw] != CZERO) {
                        kount++;
                    }
                } else {
                    if (de[(j-1) + i*ldde] != CZERO) {
                        kount++;
                    }
                    if (vw[(j-1) + i*ldvw] != CZERO) {
                        kount++;
                    }
                }
                sum += (f64)(kount - ks) * dwork[(j-1)+kw1];
            }
            dwork[(i-1)+kw2] = (f64)kount * dwork[(i-1)+kw1] + sum;
        }

        for (j = *ilo; j <= n; j++) {
            kount = 0;
            sum = ZERO;
            for (i = *ilo; i <= n; i++) {
                ks = kount;
                if (a[(i-1) + (j-1)*lda] != CZERO) {
                    kount++;
                }
                if (c[(i-1) + (j-1)*ldc] != CZERO) {
                    kount++;
                }
                sum += (f64)(kount - ks) * dwork[(i-1)+kw1];

                ks = kount;
                if (j >= i) {
                    if (de[(j-1) + (i-1)*ldde] != CZERO) {
                        kount++;
                    }
                    if (vw[(j-1) + (i-1)*ldvw] != CZERO) {
                        kount++;
                    }
                } else {
                    if (de[(i-1) + (j-1)*ldde] != CZERO) {
                        kount++;
                    }
                    if (vw[(i-1) + (j-1)*ldvw] != CZERO) {
                        kount++;
                    }
                }
                sum += (f64)(kount - ks) * dwork[(i-1)];
            }
            dwork[(j-1)+kw3] = (f64)kount * dwork[(j-1)] + sum;
        }

        sum = (SLC_DDOT(&nr, &dwork[*ilo-1+kw1], &int1, &dwork[*ilo-1+kw2], &int1) +
               SLC_DDOT(&nr, &dwork[*ilo-1], &int1, &dwork[*ilo-1+kw3], &int1)) * TWO;
        alpha = gamma / sum;

        // Determine correction to current iteration
        cmax = ZERO;
        for (i = *ilo; i <= n; i++) {
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
                goto label270;
            }
        }

        // End generalized conjugate gradient iteration
    label350:
        // Compute diagonal scaling matrices
        for (i = *ilo; i <= n; i++) {
            irab = SLC_IZAMAX(&nr, &a[(i-1) + (*ilo-1)*lda], &lda);
            rab = cabs(a[(i-1) + (*ilo-1+irab-1)*lda]);
            irab = SLC_IZAMAX(&nr, &c[(i-1) + (*ilo-1)*ldc], &ldc);
            rab = (rab > cabs(c[(i-1) + (*ilo-1+irab-1)*ldc])) ?
                   rab : cabs(c[(i-1) + (*ilo-1+irab-1)*ldc]);
            irab = SLC_IZAMAX(&i, &de[i*ldde], &int1);
            rab = (rab > cabs(de[(irab-1) + i*ldde])) ? rab : cabs(de[(irab-1) + i*ldde]);
            if (n > i) {
                i32 len_ni = n - i;
                irab = SLC_IZAMAX(&len_ni, &de[(i-1) + (i+1)*ldde], &ldde);
                rab = (rab > cabs(de[(i-1) + (i+irab)*ldde])) ?
                       rab : cabs(de[(i-1) + (i+irab)*ldde]);
            }
            irab = SLC_IZAMAX(&i, &vw[i*ldvw], &int1);
            rab = (rab > cabs(vw[(irab-1) + i*ldvw])) ? rab : cabs(vw[(irab-1) + i*ldvw]);
            if (n > i) {
                i32 len_ni = n - i;
                irab = SLC_IZAMAX(&len_ni, &vw[(i-1) + (i+1)*ldvw], &ldvw);
                rab = (rab > cabs(vw[(i-1) + (i+irab)*ldvw])) ?
                       rab : cabs(vw[(i-1) + (i+irab)*ldvw]);
            }

            lrab = (i32)(log10(rab + sfmin) / basl + ONE);
            ir = (i32)(lscale[i-1] + copysign(HALF, lscale[i-1]));
            ir = (ir < lsfmin) ? lsfmin : ir;
            ir = (ir > lsfmax) ? lsfmax : ir;
            ir = (ir > lsfmax - lrab) ? lsfmax - lrab : ir;
            lscale[i-1] = pow(SCLFAC, ir);

            icab = SLC_IZAMAX(&n, &a[(i-1)*lda], &int1);
            cab = cabs(a[(icab-1) + (i-1)*lda]);
            icab = SLC_IZAMAX(&n, &c[(i-1)*ldc], &int1);
            cab = (cab > cabs(c[(icab-1) + (i-1)*ldc])) ?
                   cab : cabs(c[(icab-1) + (i-1)*ldc]);
            icab = SLC_IZAMAX(&i, &de[(i-1)], &ldde);
            cab = (cab > cabs(de[(i-1) + (icab-1)*ldde])) ?
                   cab : cabs(de[(i-1) + (icab-1)*ldde]);
            if (n > i) {
                i32 len_ni = n - i;
                icab = SLC_IZAMAX(&len_ni, &de[i + (i-1)*ldde], &int1);
                cab = (cab > cabs(de[(i+icab-1) + (i-1)*ldde])) ?
                       cab : cabs(de[(i+icab-1) + (i-1)*ldde]);
            }
            icab = SLC_IZAMAX(&i, &vw[(i-1)], &ldvw);
            cab = (cab > cabs(vw[(i-1) + (icab-1)*ldvw])) ?
                   cab : cabs(vw[(i-1) + (icab-1)*ldvw]);
            if (n > i) {
                i32 len_ni = n - i;
                icab = SLC_IZAMAX(&len_ni, &vw[i + (i-1)*ldvw], &int1);
                cab = (cab > cabs(vw[(i+icab-1) + (i-1)*ldvw])) ?
                       cab : cabs(vw[(i+icab-1) + (i-1)*ldvw]);
            }

            lrab = (i32)(log10(cab + sfmin) / basl + ONE);
            jc = (i32)(rscale[i-1] + copysign(HALF, rscale[i-1]));
            jc = (jc < lsfmin) ? lsfmin : jc;
            jc = (jc > lsfmax) ? lsfmax : jc;
            jc = (jc > lsfmax - lrab) ? lsfmax - lrab : jc;
            rscale[i-1] = pow(SCLFAC, jc);
        }

        // Check if all scaling factors equal 1
        bool all_ones = true;
        for (i = *ilo; i <= n; i++) {
            if (lscale[i-1] != ONE || rscale[i-1] != ONE) {
                all_ones = false;
                break;
            }
        }

        if (all_ones) {
            // Finish the procedure for all scaling factors equal to 1
            nss = ns0;
            nhs = nh0;
            ths = th0;
            goto label550;
        }

        // label380
        if (loop) {
            if (ith <= -10) {
                // Compute reciprocal condition number of transformations
                ir = SLC_IDAMAX(&nr, &lscale[*ilo-1], &int1);
                jc = SLC_IDAMAX(&nr, &rscale[*ilo-1], &int1);
                t = (lscale[*ilo-1+ir-1] > rscale[*ilo-1+jc-1]) ?
                     lscale[*ilo-1+ir-1] : rscale[*ilo-1+jc-1];
                mn = t;
                for (i = *ilo; i <= n; i++) {
                    if (lscale[i-1] < mn) {
                        mn = lscale[i-1];
                    }
                }
                for (i = *ilo; i <= n; i++) {
                    if (rscale[i-1] < mn) {
                        mn = rscale[i-1];
                    }
                }
                t = mn / t;
                if (t < ONE/mxcond) {
                    th = th * TEN;
                    continue;
                } else {
                    ths = th;
                    evnorm = true;
                    goto label520;
                }
            }

            // Compute 1-norms of scaled submatrices without actually scaling
            ns = ZERO;
            for (j = *ilo; j <= n; j++) {
                t = ZERO;
                for (i = *ilo; i <= n; i++) {
                    t += cabs(a[(i-1) + (j-1)*lda]) * lscale[i-1] * rscale[j-1];
                    if (i < j) {
                        t += cabs(de[(j-1) + (i-1)*ldde]) * rscale[i-1] * rscale[j-1];
                    } else {
                        t += cabs(de[(i-1) + (j-1)*ldde]) * rscale[i-1] * rscale[j-1];
                    }
                }
                if (t > ns) {
                    ns = t;
                }
            }

            for (j = *ilo; j <= n; j++) {
                t = ZERO;
                for (i = *ilo; i <= n; i++) {
                    t += cabs(a[(j-1) + (i-1)*lda]) * lscale[j-1] * rscale[i-1];
                    if (i <= j) {
                        t += cabs(de[(i-1) + j*ldde]) * lscale[i-1] * lscale[j-1];
                    } else {
                        t += cabs(de[(j-1) + i*ldde]) * lscale[i-1] * lscale[j-1];
                    }
                }
                if (t > ns) {
                    ns = t;
                }
            }

            nh = ZERO;
            for (j = *ilo; j <= n; j++) {
                t = ZERO;
                for (i = *ilo; i <= n; i++) {
                    t += cabs(c[(i-1) + (j-1)*ldc]) * lscale[i-1] * rscale[j-1];
                    if (i < j) {
                        t += cabs(vw[(j-1) + (i-1)*ldvw]) * rscale[i-1] * rscale[j-1];
                    } else {
                        t += cabs(vw[(i-1) + (j-1)*ldvw]) * rscale[i-1] * rscale[j-1];
                    }
                }
                if (t > nh) {
                    nh = t;
                }
            }

            for (j = *ilo; j <= n; j++) {
                t = ZERO;
                for (i = *ilo; i <= n; i++) {
                    t += cabs(c[(j-1) + (i-1)*ldc]) * lscale[j-1] * rscale[i-1];
                    if (i <= j) {
                        t += cabs(vw[(i-1) + j*ldvw]) * lscale[i-1] * lscale[j-1];
                    } else {
                        t += cabs(vw[(j-1) + i*ldvw]) * lscale[i-1] * lscale[j-1];
                    }
                }
                if (t > nh) {
                    nh = t;
                }
            }

            if (ith >= -4 && ith < -2) {
                prod = (ns/denom) * (nh/denom);
                if (minpro > prod) {
                    minpro = prod;
                    stormn = true;
                    SLC_DCOPY(&nr, &lscale[*ilo-1], &int1, &dwork[kw6], &int1);
                    SLC_DCOPY(&nr, &rscale[*ilo-1], &int1, &dwork[kw7], &int1);
                    nss = ns;
                    nhs = nh;
                    ths = th;
                }
            } else if (ith >= -2) {
                if (ns < nh) {
                    ratio = (nh/ns < sfmax) ? nh/ns : sfmax;
                } else {
                    ratio = (ns/nh < sfmax) ? ns/nh : sfmax;
                }
                if (minrat > ratio) {
                    minrat = ratio;
                    stormn = true;
                    SLC_DCOPY(&nr, &lscale[*ilo-1], &int1, &dwork[kw6], &int1);
                    SLC_DCOPY(&nr, &rscale[*ilo-1], &int1, &dwork[kw7], &int1);
                    mxs = (ns > nh) ? ns : nh;
                    nss = ns;
                    nhs = nh;
                    ths = th;
                }
            }
            th = th * TEN;
        }
    }  // end for k

    // Prepare for scaling
    if (loop) {
        if (ith <= -10) {
            // Could not find well conditioned transformations
            dum = ONE;
            SLC_DCOPY(&nr, &dum, &int0, &lscale[*ilo-1], &int1);
            SLC_DCOPY(&nr, &dum, &int0, &rscale[*ilo-1], &int1);
            *iwarn = 1;
            goto label550;
        }

        // Check if scaling might reduce accuracy
        if ((mxnorm < mxs && mxnorm < mxs/MXGAIN && ith == -2) || ith == -4) {
            ir = SLC_IDAMAX(&nr, &dwork[kw6], &int1);
            jc = SLC_IDAMAX(&nr, &dwork[kw7], &int1);
            t = (dwork[kw6+ir-1] > dwork[kw7+jc-1]) ? dwork[kw6+ir-1] : dwork[kw7+jc-1];
            mn = t;
            for (i = kw6; i < kw6+nr; i++) {
                if (dwork[i] < mn) {
                    mn = dwork[i];
                }
            }
            for (i = kw7; i < kw7+nr; i++) {
                if (dwork[i] < mn) {
                    mn = dwork[i];
                }
            }
            t = mn / t;
            if (t < ONE/mxcond) {
                dum = ONE;
                SLC_DCOPY(&nr, &dum, &int0, &lscale[*ilo-1], &int1);
                SLC_DCOPY(&nr, &dum, &int0, &rscale[*ilo-1], &int1);
                *iwarn = 1;
                nss = ns0;
                nhs = nh0;
                ths = th0;
                goto label550;
            }
        }
        if (stormn) {
            SLC_DCOPY(&nr, &dwork[kw6], &int1, &lscale[*ilo-1], &int1);
            SLC_DCOPY(&nr, &dwork[kw7], &int1, &rscale[*ilo-1], &int1);
        } else {
            nss = ns;
            nhs = nh;
            ths = th;
        }
    }

label520:
    // Row scaling
    for (i = *ilo; i <= n; i++) {
        SLC_ZDSCAL(&nr, &lscale[i-1], &a[(i-1) + (*ilo-1)*lda], &lda);
        SLC_ZDSCAL(&nr, &lscale[i-1], &c[(i-1) + (*ilo-1)*ldc], &ldc);
        SLC_ZDSCAL(&i, &lscale[i-1], &de[i*ldde], &int1);
        i32 len = n - i + 1;
        SLC_ZDSCAL(&len, &lscale[i-1], &de[(i-1) + i*ldde], &ldde);
        SLC_ZDSCAL(&i, &lscale[i-1], &vw[i*ldvw], &int1);
        SLC_ZDSCAL(&len, &lscale[i-1], &vw[(i-1) + i*ldvw], &ldvw);
    }

    // Column scaling
    for (j = *ilo; j <= n; j++) {
        SLC_ZDSCAL(&n, &rscale[j-1], &a[(j-1)*lda], &int1);
        SLC_ZDSCAL(&n, &rscale[j-1], &c[(j-1)*ldc], &int1);
        SLC_ZDSCAL(&j, &rscale[j-1], &de[(j-1)], &ldde);
        i32 len = n - j + 1;
        SLC_ZDSCAL(&len, &rscale[j-1], &de[(j-1) + (j-1)*ldde], &int1);
        SLC_ZDSCAL(&j, &rscale[j-1], &vw[(j-1)], &ldvw);
        SLC_ZDSCAL(&len, &rscale[j-1], &vw[(j-1) + (j-1)*ldvw], &int1);
    }

    // Set DWORK(1:5)
label550:
    if (evnorm) {
        nss = ma02iz("skew-Hamiltonian", "1-norm", nr, &a[(*ilo-1) + (*ilo-1)*lda], lda,
                     &de[(*ilo-1) + (*ilo-1)*ldde], ldde, dwork);
        nhs = ma02iz("Hamiltonian", "1-norm", nr, &c[(*ilo-1) + (*ilo-1)*ldc], ldc,
                     &vw[(*ilo-1) + (*ilo-1)*ldvw], ldvw, dwork);
    }

    dwork[0] = ns0;
    dwork[1] = nh0;
    dwork[2] = nss;
    dwork[3] = nhs;
    if (loop) {
        dwork[4] = ths / ((mxnorm > sfmin) ? mxnorm : sfmin);
    } else {
        dwork[4] = thresh;
    }
}
